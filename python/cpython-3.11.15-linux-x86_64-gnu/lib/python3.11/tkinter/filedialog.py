"""File selection dialog classes.

Classes:

- FileDialog
- LoadFileDialog
- SaveFileDialog

This module also presents tk common file dialogues, it provides interfaces
to the native file dialogues available in Tk 4.2 and newer, and the
directory dialogue available in Tk 8.3 and newer.
These interfaces were written by Fredrik Lundh, May 1997.
"""
__all__ = ["FileDialog", "LoadFileDialog", "SaveFileDialog",
           "Open", "SaveAs", "Directory",
           "askopenfilename", "asksaveasfilename", "askopenfilenames",
           "askopenfile", "askopenfiles", "asksaveasfile", "askdirectory"]

import fnmatch
import os
from tkinter import (
    Frame, LEFT, YES, BOTTOM, Entry, TOP, Button, Tk, X,
    Toplevel, RIGHT, Y, END, Listbox, BOTH, Scrollbar,
)
from tkinter.dialog import Dialog
from tkinter import commondialog
from tkinter.simpledialog import _setup_dialog


dialogstates = {}


class FileDialog:

    """Standard file selection dialog -- no checks on selected file.

    Usage:

        d = FileDialog(master)
        fname = d.go(dir_or_file, pattern, default, key)
        if fname is None: ...canceled...
        else: ...open file...

    All arguments to go() are optional.

    The 'key' argument specifies a key in the global dictionary
    'dialogstates', which keeps track of the values for the directory
    and pattern arguments, overriding the values passed in (it does
    not keep track of the default argument!).  If no key is specified,
    the dialog keeps no memory of previous state.  Note that memory is
    kept even when the dialog is canceled.  (All this emulates the
    behavior of the Macintosh file selection dialogs.)

    """

    title = "File Selection Dialog"

    def __init__(self, master, title=None):
        if title is None: title = self.title
        self.master = master
        self.directory = None

        self.top = Toplevel(master)
        self.top.title(title)
        self.top.iconname(title)
        _setup_dialog(self.top)

        self.botframe = Frame(self.top)
        self.botframe.pack(side=BOTTOM, fill=X)

        self.selection = Entry(self.top)
        self.selection.pack(side=BOTTOM, fill=X)
        self.selection.bind('<Return>', self.ok_event)

        self.filter = Entry(self.top)
        self.filter.pack(side=TOP, fill=X)
        self.filter.bind('<Return>', self.filter_command)

        self.midframe = Frame(self.top)
        self.midframe.pack(expand=YES, fill=BOTH)

        self.filesbar = Scrollbar(self.midframe)
        self.filesbar.pack(side=RIGHT, fill=Y)
        self.files = Listbox(self.midframe, exportselection=0,
                             yscrollcommand=(self.filesbar, 'set'))
        self.files.pack(side=RIGHT, expand=YES, fill=BOTH)
        btags = self.files.bindtags()
        self.files.bindtags(btags[1:] + btags[:1])
        self.files.bind('<ButtonRelease-1>', self.files_select_event)
        self.files.bind('<Double-ButtonRelease-1>', self.files_double_event)
        self.filesbar.config(command=(self.files, 'yview'))

        self.dirsbar = Scrollbar(self.midframe)
        self.dirsbar.pack(side=LEFT, fill=Y)
        self.dirs = Listbox(self.midframe, exportselection=0,
                            yscrollcommand=(self.dirsbar, 'set'))
        self.dirs.pack(side=LEFT, expand=YES, fill=BOTH)
        self.dirsbar.config(command=(self.dirs, 'yview'))
        btags = self.dirs.bindtags()
        self.dirs.bindtags(btags[1:] + btags[:1])
        self.dirs.bind('<ButtonRelease-1>', self.dirs_select_event)
        self.dirs.bind('<Double-ButtonRelease-1>', self.dirs_double_event)

        self.ok_button = Button(self.botframe,
                                 text="OK",
                                 command=self.ok_command)
        self.ok_button.pack(side=LEFT)
        self.filter_button = Button(self.botframe,
                                    text="Filter",
                                    command=self.filter_command)
        self.filter_button.pack(side=LEFT, expand=YES)
        self.cancel_button = Button(self.botframe,
                                    text="Cancel",
                                    command=self.cancel_command)
        self.cancel_button.pack(side=RIGHT)

        self.top.protocol('WM_DELETE_WINDOW', self.cancel_command)
        # XXX Are the following okay for a general audience?
        self.top.bind('<Alt-w>', self.cancel_command)
        self.top.bind('<Alt-W>', self.cancel_command)

    def go(self, dir_or_file=os.curdir, pattern="*", default="", key=None):
        if key and key in dialogstates:
            self.directory, pattern = dialogstates[key]
        else:
            dir_or_file = os.path.expanduser(dir_or_file)
            if os.path.isdir(dir_or_file):
                self.directory = dir_or_file
            else:
                self.directory, default = os.path.split(dir_or_file)
        self.set_filter(self.directory, pattern)
        self.set_selection(default)
        self.filter_command()
        self.selection.focus_set()
        self.top.wait_visibility() # window needs to be visible for the grab
        self.top.grab_set()
        self.how = None
        self.master.mainloop()          # Exited by self.quit(how)
        if key:
            directory, pattern = self.get_filter()
            if self.how:
                directory = os.path.dirname(self.how)
            dialogstates[key] = directory, pattern
        self.top.destroy()
        return self.how

    def quit(self, how=None):
        self.how = how
        self.master.quit()              # Exit mainloop()

    def dirs_double_event(self, event):
        self.filter_command()

    def dirs_select_event(self, event):
        dir, pat = self.get_filter()
        subdir = self.dirs.get('active')
        dir = os.path.normpath(os.path.join(self.directory, subdir))
        self.set_filter(dir, pat)

    def files_double_event(self, event):
        self.ok_command()

    def files_select_event(self, event):
        file = self.files.get('active')
        self.set_selection(file)

    def ok_event(self, event):
        self.ok_command()

    def ok_command(self):
        self.quit(self.get_selection())

    def filter_command(self, event=None):
        dir, pat = self.get_filter()
        try:
            names = os.listdir(dir)
        except OSError:
            self.master.bell()
            return
        self.directory = dir
        self.set_filter(dir, pat)
        names.sort()
        subdirs = [os.pardir]
        matchingfiles = []
        for name in names:
            fullname = os.path.join(dir, name)
            if os.path.isdir(fullname):
                subdirs.append(name)
            elif fnmatch.fnmatch(name, pat):
                matchingfiles.append(name)
        self.dirs.delete(0, END)
        for name in subdirs:
            self.dirs.insert(END, name)
        self.files.delete(0, END)
        for name in matchingfiles:
            self.files.insert(END, name)
        head, tail = os.path.split(self.get_selection())
        if tail == os.curdir: tail = ''
        self.set_selection(tail)

    def get_filter(self):
        filter = self.filter.get()
        filter = os.path.expanduser(filter)
        if filter[-1:] == os.sep or os.path.isdir(filter):
            filter = os.path.join(filter, "*")
        return os.path.split(filter)

    def get_selection(self):
        file = self.selection.get()
        file = os.path.expanduser(file)
        return file

    def cancel_command(self, event=None):
        self.quit()

    def set_filter(self, dir, pat):
        if not os.path.isabs(dir):
            try:
                pwd = os.getcwd()
            except OSError:
                pwd = None
            if pwd:
                dir = os.path.join(pwd, dir)
                dir = os.path.normpath(dir)
        self.filter.delete(0, END)
        self.filter.insert(END, os.path.join(dir or os.curdir, pat or "*"))

    def set_selection(self, file):
        self.selection.delete(0, END)
        self.selection.insert(END, os.path.join(self.directory, file))


class LoadFileDialog(FileDialog):

    """File selection dialog which checks that the file exists."""

    title = "Load File Selection Dialog"

    def ok_command(self):
        file = self.get_selection()
        if not os.path.isfile(file):
            self.master.bell()
        else:
            self.quit(file)


class SaveFileDialog(FileDialog):

    """File selection dialog which checks that the file may be created."""

    title = "Save File Selection Dialog"

    def ok_command(self):
        file = self.get_selection()
        if os.path.exists(file):
            if os.path.isdir(file):
                self.master.bell()
                return
            d = Dialog(self.top,
                       title="Overwrite Existing File Question",
                       text="Overwrite existing file %r?" % (file,),
                       bitmap='questhead',
                       default=1,
                       strings=("Yes", "Cancel"))
            if d.num != 0:
                return
        else:
            head, tail = os.path.split(file)
            if not os.path.isdir(head):
                self.master.bell()
                return
        self.quit(file)


# For the following classes and modules:
#
# options (all have default values):
#
# - defaultextension: added to filename if not explicitly given
#
# - filetypes: sequence of (label, pattern) tuples.  the same pattern
#   may occur with several patterns.  use "*" as pattern to indicate
#   all files.
#
# - initialdir: initial directory.  preserved by dialog instance.
#
# - initialfile: initial file (ignored by the open dialog).  preserved
#   by dialog instance.
#
# - parent: which window to place the dialog on top of
#
# - title: dialog title
#
# - multiple: if true user may select more than one file
#
# options for the directory chooser:
#
# - initialdir, parent, title: see above
#
# - mustexist: if true, user must pick an existing directory
#


class _Dialog(commondialog.Dialog):

    def _fixoptions(self):
        try:
            # make sure "filetypes" is a tuple
            self.options["filetypes"] = tuple(self.options["filetypes"])
        except KeyError:
            pass

    def _fixresult(self, widget, result):
        if result:
            # keep directory and filename until next time
            # convert Tcl path objects to strings
            try:
                result = result.string
            except AttributeError:
                # it already is a string
                pass
            path, file = os.path.split(result)
            self.options["initialdir"] = path
            self.options["initialfile"] = file
        self.filename = result # compatibility
        return result


#
# file dialogs

class Open(_Dialog):
    "Ask for a filename to open"

    command = "tk_getOpenFile"

    def _fixresult(self, widget, result):
        if isinstance(result, tuple):
            # multiple results:
            result = tuple([getattr(r, "string", r) for r in result])
            if result:
                path, file = os.path.split(result[0])
                self.options["initialdir"] = path
                # don't set initialfile or filename, as we have multiple of these
            return result
        if not widget.tk.wantobjects() and "multiple" in self.options:
            # Need to split result explicitly
            return self._fixresult(widget, widget.tk.splitlist(result))
        return _Dialog._fixresult(self, widget, result)


class SaveAs(_Dialog):
    "Ask for a filename to save as"

    command = "tk_getSaveFile"


# the directory dialog has its own _fix routines.
class Directory(commondialog.Dialog):
    "Ask for a directory"

    command = "tk_chooseDirectory"

    def _fixresult(self, widget, result):
        if result:
            # convert Tcl path objects to strings
            try:
                result = result.string
            except AttributeError:
                # it already is a string
                pass
            # keep directory until next time
            self.options["initialdir"] = result
        self.directory = result # compatibility
        return result

#
# convenience stuff


def askopenfilename(**options):
    "Ask for a filename to open"

    return Open(**options).show()


def asksaveasfilename(**options):
    "Ask for a filename to save as"

    return SaveAs(**options).show()


def askopenfilenames(**options):
    """Ask for multiple filenames to open

    Returns a list of filenames or empty list if
    cancel button selected
    """
    options["multiple"]=1
    return Open(**options).show()

# FIXME: are the following  perhaps a bit too convenient?


def askopenfile(mode = "r", **options):
    "Ask for a filename to open, and returned the opened file"

    filename = Open(**options).show()
    if filename:
        return open(filename, mode)
    return None


def askopenfiles(mode = "r", **options):
    """Ask for multiple filenames and return the open file
    objects

    returns a list of open file objects or an empty list if
    cancel selected
    """

    files = askopenfilenames(**options)
    if files:
        ofiles=[]
        for filename in files:
            ofiles.append(open(filename, mode))
        files=ofiles
    return files


def asksaveasfile(mode = "w", **options):
    "Ask for a filename to save as, and returned the opened file"

    filename = SaveAs(**options).show()
    if filename:
        return open(filename, mode)
    return None


def askdirectory (**options):
    "Ask for a directory, and return the file name"
    return Directory(**options).show()


# --------------------------------------------------------------------
# test stuff

def test():
    """Simple test program."""
    root = Tk()
    root.withdraw()
    fd = LoadFileDialog(root)
    loadfile = fd.go(key="test")
    fd = SaveFileDialog(root)
    savefile = fd.go(key="test")
    print(loadfile, savefile)

    # Since the file name may contain non-ASCII characters, we need
    # to find an encoding that likely supports the file name, and
    # displays correctly on the terminal.

    # Start off with UTF-8
    enc = "utf-8"
    import sys

    # See whether CODESET is defined
    try:
        import locale
        locale.setlocale(locale.LC_ALL,'')
        enc = locale.nl_langinfo(locale.CODESET)
    except (ImportError, AttributeError):
        pass

    # dialog for opening files

    openfilename=askopenfilename(filetypes=[("all files", "*")])
    try:
        fp=open(openfilename,"r")
        fp.close()
    except:
        print("Could not open File: ")
        print(sys.exc_info()[1])

    print("open", openfilename.encode(enc))

    # dialog for saving files

    saveasfilename=asksaveasfilename()
    print("saveas", saveasfilename.encode(enc))


if __name__ == '__main__':
    test()
