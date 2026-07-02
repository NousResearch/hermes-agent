"""Debug user code with a GUI interface to a subclass of bdb.Bdb.

The Idb idb and Debugger gui instances each need a reference to each
other or to an rpc proxy for each other.

If IDLE is started with '-n', so that user code and idb both run in the
IDLE process, Debugger is called without an idb.  Debugger.__init__
calls Idb with its incomplete self.  Idb.__init__ stores gui and gui
then stores idb.

If IDLE is started normally, so that user code executes in a separate
process, debugger_r.start_remote_debugger is called, executing in the
IDLE process.  It calls 'start the debugger' in the remote process,
which calls Idb with a gui proxy.  Then Debugger is called in the IDLE
for more.
"""

import bdb
import os

from tkinter import *
from tkinter.ttk import Frame, Scrollbar

from idlelib import macosx
from idlelib.scrolledlist import ScrolledList
from idlelib.window import ListedToplevel


class Idb(bdb.Bdb):
    "Supply user_line and user_exception functions for Bdb."

    def __init__(self, gui):
        self.gui = gui  # An instance of Debugger or proxy thereof.
        super().__init__()

    def user_line(self, frame):
        """Handle a user stopping or breaking at a line.

        Convert frame to a string and send it to gui.
        """
        if _in_rpc_code(frame):
            self.set_step()
            return
        message = _frame2message(frame)
        try:
            self.gui.interaction(message, frame)
        except TclError:  # When closing debugger window with [x] in 3.x
            pass

    def user_exception(self, frame, exc_info):
        """Handle an the occurrence of an exception."""
        if _in_rpc_code(frame):
            self.set_step()
            return
        message = _frame2message(frame)
        self.gui.interaction(message, frame, exc_info)

def _in_rpc_code(frame):
    "Determine if debugger is within RPC code."
    if frame.f_code.co_filename.count('rpc.py'):
        return True  # Skip this frame.
    else:
        prev_frame = frame.f_back
        if prev_frame is None:
            return False
        prev_name = prev_frame.f_code.co_filename
        if 'idlelib' in prev_name and 'debugger' in prev_name:
            # catch both idlelib/debugger.py and idlelib/debugger_r.py
            # on both Posix and Windows
            return False
        return _in_rpc_code(prev_frame)

def _frame2message(frame):
    """Return a message string for frame."""
    code = frame.f_code
    filename = code.co_filename
    lineno = frame.f_lineno
    basename = os.path.basename(filename)
    message = f"{basename}:{lineno}"
    if code.co_name != "?":
        message = f"{message}: {code.co_name}()"
    return message


class Debugger:
    """The debugger interface.

    This class handles the drawing of the debugger window and
    the interactions with the underlying debugger session.
    """
    vstack = None
    vsource = None
    vlocals = None
    vglobals = None
    stackviewer = None
    localsviewer = None
    globalsviewer = None

    def __init__(self, pyshell, idb=None):
        """Instantiate and draw a debugger window.

        :param pyshell: An instance of the PyShell Window
        :type  pyshell: :class:`idlelib.pyshell.PyShell`

        :param idb: An instance of the IDLE debugger (optional)
        :type  idb: :class:`idlelib.debugger.Idb`
        """
        if idb is None:
            idb = Idb(self)
        self.pyshell = pyshell
        self.idb = idb  # If passed, a proxy of remote instance.
        self.frame = None
        self.make_gui()
        self.interacting = False
        self.nesting_level = 0

    def run(self, *args):
        """Run the debugger."""
        # Deal with the scenario where we've already got a program running
        # in the debugger and we want to start another. If that is the case,
        # our second 'run' was invoked from an event dispatched not from
        # the main event loop, but from the nested event loop in 'interaction'
        # below. So our stack looks something like this:
        #       outer main event loop
        #         run()
        #           <running program with traces>
        #             callback to debugger's interaction()
        #               nested event loop
        #                 run() for second command
        #
        # This kind of nesting of event loops causes all kinds of problems
        # (see e.g. issue #24455) especially when dealing with running as a
        # subprocess, where there's all kinds of extra stuff happening in
        # there - insert a traceback.print_stack() to check it out.
        #
        # By this point, we've already called restart_subprocess() in
        # ScriptBinding. However, we also need to unwind the stack back to
        # that outer event loop.  To accomplish this, we:
        #   - return immediately from the nested run()
        #   - abort_loop ensures the nested event loop will terminate
        #   - the debugger's interaction routine completes normally
        #   - the restart_subprocess() will have taken care of stopping
        #     the running program, which will also let the outer run complete
        #
        # That leaves us back at the outer main event loop, at which point our
        # after event can fire, and we'll come back to this routine with a
        # clean stack.
        if self.nesting_level > 0:
            self.abort_loop()
            self.root.after(100, lambda: self.run(*args))
            return
        try:
            self.interacting = True
            return self.idb.run(*args)
        finally:
            self.interacting = False

    def close(self, event=None):
        """Close the debugger and window."""
        try:
            self.quit()
        except Exception:
            pass
        if self.interacting:
            self.top.bell()
            return
        if self.stackviewer:
            self.stackviewer.close(); self.stackviewer = None
        # Clean up pyshell if user clicked debugger control close widget.
        # (Causes a harmless extra cycle through close_debugger() if user
        # toggled debugger from pyshell Debug menu)
        self.pyshell.close_debugger()
        # Now close the debugger control window....
        self.top.destroy()

    def make_gui(self):
        """Draw the debugger gui on the screen."""
        pyshell = self.pyshell
        self.flist = pyshell.flist
        self.root = root = pyshell.root
        self.top = top = ListedToplevel(root)
        self.top.wm_title("Debug Control")
        self.top.wm_iconname("Debug")
        top.wm_protocol("WM_DELETE_WINDOW", self.close)
        self.top.bind("<Escape>", self.close)

        self.bframe = bframe = Frame(top)
        self.bframe.pack(anchor="w")
        self.buttons = bl = []

        self.bcont = b = Button(bframe, text="Go", command=self.cont)
        bl.append(b)
        self.bstep = b = Button(bframe, text="Step", command=self.step)
        bl.append(b)
        self.bnext = b = Button(bframe, text="Over", command=self.next)
        bl.append(b)
        self.bret = b = Button(bframe, text="Out", command=self.ret)
        bl.append(b)
        self.bret = b = Button(bframe, text="Quit", command=self.quit)
        bl.append(b)

        for b in bl:
            b.configure(state="disabled")
            b.pack(side="left")

        self.cframe = cframe = Frame(bframe)
        self.cframe.pack(side="left")

        if not self.vstack:
            self.__class__.vstack = BooleanVar(top)
            self.vstack.set(1)
        self.bstack = Checkbutton(cframe,
            text="Stack", command=self.show_stack, variable=self.vstack)
        self.bstack.grid(row=0, column=0)
        if not self.vsource:
            self.__class__.vsource = BooleanVar(top)
        self.bsource = Checkbutton(cframe,
            text="Source", command=self.show_source, variable=self.vsource)
        self.bsource.grid(row=0, column=1)
        if not self.vlocals:
            self.__class__.vlocals = BooleanVar(top)
            self.vlocals.set(1)
        self.blocals = Checkbutton(cframe,
            text="Locals", command=self.show_locals, variable=self.vlocals)
        self.blocals.grid(row=1, column=0)
        if not self.vglobals:
            self.__class__.vglobals = BooleanVar(top)
        self.bglobals = Checkbutton(cframe,
            text="Globals", command=self.show_globals, variable=self.vglobals)
        self.bglobals.grid(row=1, column=1)

        self.status = Label(top, anchor="w")
        self.status.pack(anchor="w")
        self.error = Label(top, anchor="w")
        self.error.pack(anchor="w", fill="x")
        self.errorbg = self.error.cget("background")

        self.fstack = Frame(top, height=1)
        self.fstack.pack(expand=1, fill="both")
        self.flocals = Frame(top)
        self.flocals.pack(expand=1, fill="both")
        self.fglobals = Frame(top, height=1)
        self.fglobals.pack(expand=1, fill="both")

        if self.vstack.get():
            self.show_stack()
        if self.vlocals.get():
            self.show_locals()
        if self.vglobals.get():
            self.show_globals()

    def interaction(self, message, frame, info=None):
        self.frame = frame
        self.status.configure(text=message)

        if info:
            type, value, tb = info
            try:
                m1 = type.__name__
            except AttributeError:
                m1 = "%s" % str(type)
            if value is not None:
                try:
                   # TODO redo entire section, tries not needed.
                    m1 = f"{m1}: {value}"
                except:
                    pass
            bg = "yellow"
        else:
            m1 = ""
            tb = None
            bg = self.errorbg
        self.error.configure(text=m1, background=bg)

        sv = self.stackviewer
        if sv:
            stack, i = self.idb.get_stack(self.frame, tb)
            sv.load_stack(stack, i)

        self.show_variables(1)

        if self.vsource.get():
            self.sync_source_line()

        for b in self.buttons:
            b.configure(state="normal")

        self.top.wakeup()
        # Nested main loop: Tkinter's main loop is not reentrant, so use
        # Tcl's vwait facility, which reenters the event loop until an
        # event handler sets the variable we're waiting on.
        self.nesting_level += 1
        self.root.tk.call('vwait', '::idledebugwait')
        self.nesting_level -= 1

        for b in self.buttons:
            b.configure(state="disabled")
        self.status.configure(text="")
        self.error.configure(text="", background=self.errorbg)
        self.frame = None

    def sync_source_line(self):
        frame = self.frame
        if not frame:
            return
        filename, lineno = self.__frame2fileline(frame)
        if filename[:1] + filename[-1:] != "<>" and os.path.exists(filename):
            self.flist.gotofileline(filename, lineno)

    def __frame2fileline(self, frame):
        code = frame.f_code
        filename = code.co_filename
        lineno = frame.f_lineno
        return filename, lineno

    def cont(self):
        self.idb.set_continue()
        self.abort_loop()

    def step(self):
        self.idb.set_step()
        self.abort_loop()

    def next(self):
        self.idb.set_next(self.frame)
        self.abort_loop()

    def ret(self):
        self.idb.set_return(self.frame)
        self.abort_loop()

    def quit(self):
        self.idb.set_quit()
        self.abort_loop()

    def abort_loop(self):
        self.root.tk.call('set', '::idledebugwait', '1')

    def show_stack(self):
        if not self.stackviewer and self.vstack.get():
            self.stackviewer = sv = StackViewer(self.fstack, self.flist, self)
            if self.frame:
                stack, i = self.idb.get_stack(self.frame, None)
                sv.load_stack(stack, i)
        else:
            sv = self.stackviewer
            if sv and not self.vstack.get():
                self.stackviewer = None
                sv.close()
            self.fstack['height'] = 1

    def show_source(self):
        if self.vsource.get():
            self.sync_source_line()

    def show_frame(self, stackitem):
        self.frame = stackitem[0]  # lineno is stackitem[1]
        self.show_variables()

    def show_locals(self):
        lv = self.localsviewer
        if self.vlocals.get():
            if not lv:
                self.localsviewer = NamespaceViewer(self.flocals, "Locals")
        else:
            if lv:
                self.localsviewer = None
                lv.close()
                self.flocals['height'] = 1
        self.show_variables()

    def show_globals(self):
        gv = self.globalsviewer
        if self.vglobals.get():
            if not gv:
                self.globalsviewer = NamespaceViewer(self.fglobals, "Globals")
        else:
            if gv:
                self.globalsviewer = None
                gv.close()
                self.fglobals['height'] = 1
        self.show_variables()

    def show_variables(self, force=0):
        lv = self.localsviewer
        gv = self.globalsviewer
        frame = self.frame
        if not frame:
            ldict = gdict = None
        else:
            ldict = frame.f_locals
            gdict = frame.f_globals
            if lv and gv and ldict is gdict:
                ldict = None
        if lv:
            lv.load_dict(ldict, force, self.pyshell.interp.rpcclt)
        if gv:
            gv.load_dict(gdict, force, self.pyshell.interp.rpcclt)

    def set_breakpoint(self, filename, lineno):
        """Set a filename-lineno breakpoint in the debugger.

        Called from self.load_breakpoints and EW.setbreakpoint
        """
        self.idb.set_break(filename, lineno)

    def clear_breakpoint(self, filename, lineno):
        self.idb.clear_break(filename, lineno)

    def clear_file_breaks(self, filename):
        self.idb.clear_all_file_breaks(filename)

    def load_breakpoints(self):
        """Load PyShellEditorWindow breakpoints into subprocess debugger."""
        for editwin in self.pyshell.flist.inversedict:
            filename = editwin.io.filename
            try:
                for lineno in editwin.breakpoints:
                    self.set_breakpoint(filename, lineno)
            except AttributeError:
                continue


class StackViewer(ScrolledList):
    "Code stack viewer for debugger GUI."

    def __init__(self, master, flist, gui):
        if macosx.isAquaTk():
            # At least on with the stock AquaTk version on OSX 10.4 you'll
            # get a shaking GUI that eventually kills IDLE if the width
            # argument is specified.
            ScrolledList.__init__(self, master)
        else:
            ScrolledList.__init__(self, master, width=80)
        self.flist = flist
        self.gui = gui
        self.stack = []

    def load_stack(self, stack, index=None):
        self.stack = stack
        self.clear()
        for i in range(len(stack)):
            frame, lineno = stack[i]
            try:
                modname = frame.f_globals["__name__"]
            except:
                modname = "?"
            code = frame.f_code
            filename = code.co_filename
            funcname = code.co_name
            import linecache
            sourceline = linecache.getline(filename, lineno)
            sourceline = sourceline.strip()
            if funcname in ("?", "", None):
                item = "%s, line %d: %s" % (modname, lineno, sourceline)
            else:
                item = "%s.%s(), line %d: %s" % (modname, funcname,
                                                 lineno, sourceline)
            if i == index:
                item = "> " + item
            self.append(item)
        if index is not None:
            self.select(index)

    def popup_event(self, event):
        "Override base method."
        if self.stack:
            return ScrolledList.popup_event(self, event)

    def fill_menu(self):
        "Override base method."
        menu = self.menu
        menu.add_command(label="Go to source line",
                         command=self.goto_source_line)
        menu.add_command(label="Show stack frame",
                         command=self.show_stack_frame)

    def on_select(self, index):
        "Override base method."
        if 0 <= index < len(self.stack):
            self.gui.show_frame(self.stack[index])

    def on_double(self, index):
        "Override base method."
        self.show_source(index)

    def goto_source_line(self):
        index = self.listbox.index("active")
        self.show_source(index)

    def show_stack_frame(self):
        index = self.listbox.index("active")
        if 0 <= index < len(self.stack):
            self.gui.show_frame(self.stack[index])

    def show_source(self, index):
        if not (0 <= index < len(self.stack)):
            return
        frame, lineno = self.stack[index]
        code = frame.f_code
        filename = code.co_filename
        if os.path.isfile(filename):
            edit = self.flist.open(filename)
            if edit:
                edit.gotoline(lineno)


class NamespaceViewer:
    "Global/local namespace viewer for debugger GUI."

    def __init__(self, master, title, odict=None):  # XXX odict never passed.
        width = 0
        height = 40
        if odict:
            height = 20*len(odict) # XXX 20 == observed height of Entry widget
        self.master = master
        self.title = title
        import reprlib
        self.repr = reprlib.Repr()
        self.repr.maxstring = 60
        self.repr.maxother = 60
        self.frame = frame = Frame(master)
        self.frame.pack(expand=1, fill="both")
        self.label = Label(frame, text=title, borderwidth=2, relief="groove")
        self.label.pack(fill="x")
        self.vbar = vbar = Scrollbar(frame, name="vbar")
        vbar.pack(side="right", fill="y")
        self.canvas = canvas = Canvas(frame,
                                      height=min(300, max(40, height)),
                                      scrollregion=(0, 0, width, height))
        canvas.pack(side="left", fill="both", expand=1)
        vbar["command"] = canvas.yview
        canvas["yscrollcommand"] = vbar.set
        self.subframe = subframe = Frame(canvas)
        self.sfid = canvas.create_window(0, 0, window=subframe, anchor="nw")
        self.load_dict(odict)

    prev_odict = -1  # Needed for initial comparison below.

    def load_dict(self, odict, force=0, rpc_client=None):
        if odict is self.prev_odict and not force:
            return
        subframe = self.subframe
        frame = self.frame
        for c in list(subframe.children.values()):
            c.destroy()
        self.prev_odict = None
        if not odict:
            l = Label(subframe, text="None")
            l.grid(row=0, column=0)
        else:
            #names = sorted(dict)
            #
            # Because of (temporary) limitations on the dict_keys type (not yet
            # public or pickleable), have the subprocess to send a list of
            # keys, not a dict_keys object.  sorted() will take a dict_keys
            # (no subprocess) or a list.
            #
            # There is also an obscure bug in sorted(dict) where the
            # interpreter gets into a loop requesting non-existing dict[0],
            # dict[1], dict[2], etc from the debugger_r.DictProxy.
            # TODO recheck above; see debugger_r 159ff, debugobj 60.
            keys_list = odict.keys()
            names = sorted(keys_list)

            row = 0
            for name in names:
                value = odict[name]
                svalue = self.repr.repr(value) # repr(value)
                # Strip extra quotes caused by calling repr on the (already)
                # repr'd value sent across the RPC interface:
                if rpc_client:
                    svalue = svalue[1:-1]
                l = Label(subframe, text=name)
                l.grid(row=row, column=0, sticky="nw")
                l = Entry(subframe, width=0, borderwidth=0)
                l.insert(0, svalue)
                l.grid(row=row, column=1, sticky="nw")
                row = row+1
        self.prev_odict = odict
        # XXX Could we use a <Configure> callback for the following?
        subframe.update_idletasks() # Alas!
        width = subframe.winfo_reqwidth()
        height = subframe.winfo_reqheight()
        canvas = self.canvas
        self.canvas["scrollregion"] = (0, 0, width, height)
        if height > 300:
            canvas["height"] = 300
            frame.pack(expand=1)
        else:
            canvas["height"] = height
            frame.pack(expand=0)

    def close(self):
        self.frame.destroy()


if __name__ == "__main__":
    from unittest import main
    main('idlelib.idle_test.test_debugger', verbosity=2, exit=False)

# TODO: htest?
