import string

from idlelib.delegator import Delegator

# tkinter import not needed because module does not create widgets,
# although many methods operate on text widget arguments.

#$ event <<redo>>
#$ win <Control-y>
#$ unix <Alt-z>

#$ event <<undo>>
#$ win <Control-z>
#$ unix <Control-z>

#$ event <<dump-undo-state>>
#$ win <Control-backslash>
#$ unix <Control-backslash>


class UndoDelegator(Delegator):

    max_undo = 1000

    def __init__(self):
        Delegator.__init__(self)
        self.reset_undo()

    def setdelegate(self, delegate):
        if self.delegate is not None:
            self.unbind("<<undo>>")
            self.unbind("<<redo>>")
            self.unbind("<<dump-undo-state>>")
        Delegator.setdelegate(self, delegate)
        if delegate is not None:
            self.bind("<<undo>>", self.undo_event)
            self.bind("<<redo>>", self.redo_event)
            self.bind("<<dump-undo-state>>", self.dump_event)

    def dump_event(self, event):
        from pprint import pprint
        pprint(self.undolist[:self.pointer])
        print("pointer:", self.pointer, end=' ')
        print("saved:", self.saved, end=' ')
        print("can_merge:", self.can_merge, end=' ')
        print("get_saved():", self.get_saved())
        pprint(self.undolist[self.pointer:])
        return "break"

    def reset_undo(self):
        self.was_saved = -1
        self.pointer = 0
        self.undolist = []
        self.undoblock = 0  # or a CommandSequence instance
        self.set_saved(1)

    def set_saved(self, flag):
        if flag:
            self.saved = self.pointer
        else:
            self.saved = -1
        self.can_merge = False
        self.check_saved()

    def get_saved(self):
        return self.saved == self.pointer

    saved_change_hook = None

    def set_saved_change_hook(self, hook):
        self.saved_change_hook = hook

    was_saved = -1

    def check_saved(self):
        is_saved = self.get_saved()
        if is_saved != self.was_saved:
            self.was_saved = is_saved
            if self.saved_change_hook:
                self.saved_change_hook()

    def insert(self, index, chars, tags=None):
        self.addcmd(InsertCommand(index, chars, tags))

    def delete(self, index1, index2=None):
        self.addcmd(DeleteCommand(index1, index2))

    # Clients should call undo_block_start() and undo_block_stop()
    # around a sequence of editing cmds to be treated as a unit by
    # undo & redo.  Nested matching calls are OK, and the inner calls
    # then act like nops.  OK too if no editing cmds, or only one
    # editing cmd, is issued in between:  if no cmds, the whole
    # sequence has no effect; and if only one cmd, that cmd is entered
    # directly into the undo list, as if undo_block_xxx hadn't been
    # called.  The intent of all that is to make this scheme easy
    # to use:  all the client has to worry about is making sure each
    # _start() call is matched by a _stop() call.

    def undo_block_start(self):
        if self.undoblock == 0:
            self.undoblock = CommandSequence()
        self.undoblock.bump_depth()

    def undo_block_stop(self):
        if self.undoblock.bump_depth(-1) == 0:
            cmd = self.undoblock
            self.undoblock = 0
            if len(cmd) > 0:
                if len(cmd) == 1:
                    # no need to wrap a single cmd
                    cmd = cmd.getcmd(0)
                # this blk of cmds, or single cmd, has already
                # been done, so don't execute it again
                self.addcmd(cmd, 0)

    def addcmd(self, cmd, execute=True):
        if execute:
            cmd.do(self.delegate)
        if self.undoblock != 0:
            self.undoblock.append(cmd)
            return
        if self.can_merge and self.pointer > 0:
            lastcmd = self.undolist[self.pointer-1]
            if lastcmd.merge(cmd):
                return
        self.undolist[self.pointer:] = [cmd]
        if self.saved > self.pointer:
            self.saved = -1
        self.pointer = self.pointer + 1
        if len(self.undolist) > self.max_undo:
            ##print "truncating undo list"
            del self.undolist[0]
            self.pointer = self.pointer - 1
            if self.saved >= 0:
                self.saved = self.saved - 1
        self.can_merge = True
        self.check_saved()

    def undo_event(self, event):
        if self.pointer == 0:
            self.bell()
            return "break"
        cmd = self.undolist[self.pointer - 1]
        cmd.undo(self.delegate)
        self.pointer = self.pointer - 1
        self.can_merge = False
        self.check_saved()
        return "break"

    def redo_event(self, event):
        if self.pointer >= len(self.undolist):
            self.bell()
            return "break"
        cmd = self.undolist[self.pointer]
        cmd.redo(self.delegate)
        self.pointer = self.pointer + 1
        self.can_merge = False
        self.check_saved()
        return "break"


class Command:
    # Base class for Undoable commands

    tags = None

    def __init__(self, index1, index2, chars, tags=None):
        self.marks_before = {}
        self.marks_after = {}
        self.index1 = index1
        self.index2 = index2
        self.chars = chars
        if tags:
            self.tags = tags

    def __repr__(self):
        s = self.__class__.__name__
        t = (self.index1, self.index2, self.chars, self.tags)
        if self.tags is None:
            t = t[:-1]
        return s + repr(t)

    def do(self, text):
        pass

    def redo(self, text):
        pass

    def undo(self, text):
        pass

    def merge(self, cmd):
        return 0

    def save_marks(self, text):
        marks = {}
        for name in text.mark_names():
            if name != "insert" and name != "current":
                marks[name] = text.index(name)
        return marks

    def set_marks(self, text, marks):
        for name, index in marks.items():
            text.mark_set(name, index)


class InsertCommand(Command):
    # Undoable insert command

    def __init__(self, index1, chars, tags=None):
        Command.__init__(self, index1, None, chars, tags)

    def do(self, text):
        self.marks_before = self.save_marks(text)
        self.index1 = text.index(self.index1)
        if text.compare(self.index1, ">", "end-1c"):
            # Insert before the final newline
            self.index1 = text.index("end-1c")
        text.insert(self.index1, self.chars, self.tags)
        self.index2 = text.index("%s+%dc" % (self.index1, len(self.chars)))
        self.marks_after = self.save_marks(text)
        ##sys.__stderr__.write("do: %s\n" % self)

    def redo(self, text):
        text.mark_set('insert', self.index1)
        text.insert(self.index1, self.chars, self.tags)
        self.set_marks(text, self.marks_after)
        text.see('insert')
        ##sys.__stderr__.write("redo: %s\n" % self)

    def undo(self, text):
        text.mark_set('insert', self.index1)
        text.delete(self.index1, self.index2)
        self.set_marks(text, self.marks_before)
        text.see('insert')
        ##sys.__stderr__.write("undo: %s\n" % self)

    def merge(self, cmd):
        if self.__class__ is not cmd.__class__:
            return False
        if self.index2 != cmd.index1:
            return False
        if self.tags != cmd.tags:
            return False
        if len(cmd.chars) != 1:
            return False
        if self.chars and \
           self.classify(self.chars[-1]) != self.classify(cmd.chars):
            return False
        self.index2 = cmd.index2
        self.chars = self.chars + cmd.chars
        return True

    alphanumeric = string.ascii_letters + string.digits + "_"

    def classify(self, c):
        if c in self.alphanumeric:
            return "alphanumeric"
        if c == "\n":
            return "newline"
        return "punctuation"


class DeleteCommand(Command):
    # Undoable delete command

    def __init__(self, index1, index2=None):
        Command.__init__(self, index1, index2, None, None)

    def do(self, text):
        self.marks_before = self.save_marks(text)
        self.index1 = text.index(self.index1)
        if self.index2:
            self.index2 = text.index(self.index2)
        else:
            self.index2 = text.index(self.index1 + " +1c")
        if text.compare(self.index2, ">", "end-1c"):
            # Don't delete the final newline
            self.index2 = text.index("end-1c")
        self.chars = text.get(self.index1, self.index2)
        text.delete(self.index1, self.index2)
        self.marks_after = self.save_marks(text)
        ##sys.__stderr__.write("do: %s\n" % self)

    def redo(self, text):
        text.mark_set('insert', self.index1)
        text.delete(self.index1, self.index2)
        self.set_marks(text, self.marks_after)
        text.see('insert')
        ##sys.__stderr__.write("redo: %s\n" % self)

    def undo(self, text):
        text.mark_set('insert', self.index1)
        text.insert(self.index1, self.chars)
        self.set_marks(text, self.marks_before)
        text.see('insert')
        ##sys.__stderr__.write("undo: %s\n" % self)


class CommandSequence(Command):
    # Wrapper for a sequence of undoable cmds to be undone/redone
    # as a unit

    def __init__(self):
        self.cmds = []
        self.depth = 0

    def __repr__(self):
        s = self.__class__.__name__
        strs = []
        for cmd in self.cmds:
            strs.append(f"    {cmd!r}")
        return s + "(\n" + ",\n".join(strs) + "\n)"

    def __len__(self):
        return len(self.cmds)

    def append(self, cmd):
        self.cmds.append(cmd)

    def getcmd(self, i):
        return self.cmds[i]

    def redo(self, text):
        for cmd in self.cmds:
            cmd.redo(text)

    def undo(self, text):
        cmds = self.cmds[:]
        cmds.reverse()
        for cmd in cmds:
            cmd.undo(text)

    def bump_depth(self, incr=1):
        self.depth = self.depth + incr
        return self.depth


def _undo_delegator(parent):  # htest #
    from tkinter import Toplevel, Text, Button
    from idlelib.percolator import Percolator
    top = Toplevel(parent)
    top.title("Test UndoDelegator")
    x, y = map(int, parent.geometry().split('+')[1:])
    top.geometry("+%d+%d" % (x, y + 175))

    text = Text(top, height=10)
    text.pack()
    text.focus_set()
    p = Percolator(text)
    d = UndoDelegator()
    p.insertfilter(d)

    undo = Button(top, text="Undo", command=lambda:d.undo_event(None))
    undo.pack(side='left')
    redo = Button(top, text="Redo", command=lambda:d.redo_event(None))
    redo.pack(side='left')
    dump = Button(top, text="Dump", command=lambda:d.dump_event(None))
    dump.pack(side='left')


if __name__ == "__main__":
    from unittest import main
    main('idlelib.idle_test.test_undo', verbosity=2, exit=False)

    from idlelib.idle_test.htest import run
    run(_undo_delegator)
