"""Grep dialog for Find in Files functionality.

   Inherits from SearchDialogBase for GUI and uses searchengine
   to prepare search pattern.
"""
import fnmatch
import os
import sys

from tkinter import StringVar, BooleanVar
from tkinter.ttk import Checkbutton  # Frame imported in ...Base

from idlelib.searchbase import SearchDialogBase
from idlelib import searchengine

# Importing OutputWindow here fails due to import loop
# EditorWindow -> GrepDialog -> OutputWindow -> EditorWindow


def grep(text, io=None, flist=None):
    """Open the Find in Files dialog.

    Module-level function to access the singleton GrepDialog
    instance and open the dialog.  If text is selected, it is
    used as the search phrase; otherwise, the previous entry
    is used.

    Args:
        text: Text widget that contains the selected text for
              default search phrase.
        io: iomenu.IOBinding instance with default path to search.
        flist: filelist.FileList instance for OutputWindow parent.
    """
    root = text._root()
    engine = searchengine.get(root)
    if not hasattr(engine, "_grepdialog"):
        engine._grepdialog = GrepDialog(root, engine, flist)
    dialog = engine._grepdialog
    searchphrase = text.get("sel.first", "sel.last")
    dialog.open(text, searchphrase, io)


def walk_error(msg):
    "Handle os.walk error."
    print(msg)


def findfiles(folder, pattern, recursive):
    """Generate file names in dir that match pattern.

    Args:
        folder: Root directory to search.
        pattern: File pattern to match.
        recursive: True to include subdirectories.
    """
    for dirpath, _, filenames in os.walk(folder, onerror=walk_error):
        yield from (os.path.join(dirpath, name)
                    for name in filenames
                    if fnmatch.fnmatch(name, pattern))
        if not recursive:
            break


class GrepDialog(SearchDialogBase):
    "Dialog for searching multiple files."

    title = "Find in Files Dialog"
    icon = "Grep"
    needwrapbutton = 0

    def __init__(self, root, engine, flist):
        """Create search dialog for searching for a phrase in the file system.

        Uses SearchDialogBase as the basis for the GUI and a
        searchengine instance to prepare the search.

        Attributes:
            flist: filelist.Filelist instance for OutputWindow parent.
            globvar: String value of Entry widget for path to search.
            globent: Entry widget for globvar.  Created in
                create_entries().
            recvar: Boolean value of Checkbutton widget for
                traversing through subdirectories.
        """
        super().__init__(root, engine)
        self.flist = flist
        self.globvar = StringVar(root)
        self.recvar = BooleanVar(root)

    def open(self, text, searchphrase, io=None):
        """Make dialog visible on top of others and ready to use.

        Extend the SearchDialogBase open() to set the initial value
        for globvar.

        Args:
            text: Multicall object containing the text information.
            searchphrase: String phrase to search.
            io: iomenu.IOBinding instance containing file path.
        """
        SearchDialogBase.open(self, text, searchphrase)
        if io:
            path = io.filename or ""
        else:
            path = ""
        dir, base = os.path.split(path)
        head, tail = os.path.splitext(base)
        if not tail:
            tail = ".py"
        self.globvar.set(os.path.join(dir, "*" + tail))

    def create_entries(self):
        "Create base entry widgets and add widget for search path."
        SearchDialogBase.create_entries(self)
        self.globent = self.make_entry("In files:", self.globvar)[0]

    def create_other_buttons(self):
        "Add check button to recurse down subdirectories."
        btn = Checkbutton(
                self.make_frame()[0], variable=self.recvar,
                text="Recurse down subdirectories")
        btn.pack(side="top", fill="both")

    def create_command_buttons(self):
        "Create base command buttons and add button for Search Files."
        SearchDialogBase.create_command_buttons(self)
        self.make_button("Search Files", self.default_command, isdef=True)

    def default_command(self, event=None):
        """Grep for search pattern in file path. The default command is bound
        to <Return>.

        If entry values are populated, set OutputWindow as stdout
        and perform search.  The search dialog is closed automatically
        when the search begins.
        """
        prog = self.engine.getprog()
        if not prog:
            return
        path = self.globvar.get()
        if not path:
            self.top.bell()
            return
        from idlelib.outwin import OutputWindow  # leave here!
        save = sys.stdout
        try:
            sys.stdout = OutputWindow(self.flist)
            self.grep_it(prog, path)
        finally:
            sys.stdout = save

    def grep_it(self, prog, path):
        """Search for prog within the lines of the files in path.

        For the each file in the path directory, open the file and
        search each line for the matching pattern.  If the pattern is
        found,  write the file and line information to stdout (which
        is an OutputWindow).

        Args:
            prog: The compiled, cooked search pattern.
            path: String containing the search path.
        """
        folder, filepat = os.path.split(path)
        if not folder:
            folder = os.curdir
        filelist = sorted(findfiles(folder, filepat, self.recvar.get()))
        self.close()
        pat = self.engine.getpat()
        print(f"Searching {pat!r} in {path} ...")
        hits = 0
        try:
            for fn in filelist:
                try:
                    with open(fn, errors='replace') as f:
                        for lineno, line in enumerate(f, 1):
                            if line[-1:] == '\n':
                                line = line[:-1]
                            if prog.search(line):
                                sys.stdout.write(f"{fn}: {lineno}: {line}\n")
                                hits += 1
                except OSError as msg:
                    print(msg)
            print(f"Hits found: {hits}\n(Hint: right-click to open locations.)"
                  if hits else "No hits.")
        except AttributeError:
            # Tk window has been closed, OutputWindow.text = None,
            # so in OW.write, OW.text.insert fails.
            pass


def _grep_dialog(parent):  # htest #
    from tkinter import Toplevel, Text, SEL, END
    from tkinter.ttk import Frame, Button
    from idlelib.pyshell import PyShellFileList

    top = Toplevel(parent)
    top.title("Test GrepDialog")
    x, y = map(int, parent.geometry().split('+')[1:])
    top.geometry(f"+{x}+{y + 175}")

    flist = PyShellFileList(top)
    frame = Frame(top)
    frame.pack()
    text = Text(frame, height=5)
    text.pack()
    text.insert('1.0', 'import grep')

    def show_grep_dialog():
        text.tag_add(SEL, "1.0", '1.end')
        grep(text, flist=flist)
        text.tag_remove(SEL, "1.0", '1.end')

    button = Button(frame, text="Show GrepDialog", command=show_grep_dialog)
    button.pack()


if __name__ == "__main__":
    from unittest import main
    main('idlelib.idle_test.test_grep', verbosity=2, exit=False)

    from idlelib.idle_test.htest import run
    run(_grep_dialog)
