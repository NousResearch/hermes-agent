"""
Dialogs that query users and verify the answer before accepting.

Query is the generic base class for a popup dialog.
The user must either enter a valid answer or close the dialog.
Entries are validated when <Return> is entered or [Ok] is clicked.
Entries are ignored when [Cancel] or [X] are clicked.
The 'return value' is .result set to either a valid answer or None.

Subclass SectionName gets a name for a new config file section.
Configdialog uses it for new highlight theme and keybinding set names.
Subclass ModuleName gets a name for File => Open Module.
Subclass HelpSource gets menu item and path for additions to Help menu.
"""
# Query and Section name result from splitting GetCfgSectionNameDialog
# of configSectionNameDialog.py (temporarily config_sec.py) into
# generic and specific parts.  3.6 only, July 2016.
# ModuleName.entry_ok came from editor.EditorWindow.load_module.
# HelpSource was extracted from configHelpSourceEdit.py (temporarily
# config_help.py), with darwin code moved from ok to path_ok.

import importlib.util, importlib.abc
import os
import shlex
from sys import executable, platform  # Platform is set for one test.

from tkinter import Toplevel, StringVar, BooleanVar, W, E, S
from tkinter.ttk import Frame, Button, Entry, Label, Checkbutton
from tkinter import filedialog
from tkinter.font import Font
from tkinter.simpledialog import _setup_dialog

class Query(Toplevel):
    """Base class for getting verified answer from a user.

    For this base class, accept any non-blank string.
    """
    def __init__(self, parent, title, message, *, text0='', used_names={},
                 _htest=False, _utest=False):
        """Create modal popup, return when destroyed.

        Additional subclass init must be done before this unless
        _utest=True is passed to suppress wait_window().

        title - string, title of popup dialog
        message - string, informational message to display
        text0 - initial value for entry
        used_names - names already in use
        _htest - bool, change box location when running htest
        _utest - bool, leave window hidden and not modal
        """
        self.parent = parent  # Needed for Font call.
        self.message = message
        self.text0 = text0
        self.used_names = used_names

        Toplevel.__init__(self, parent)
        self.withdraw()  # Hide while configuring, especially geometry.
        self.title(title)
        self.transient(parent)
        if not _utest:  # Otherwise fail when directly run unittest.
            self.grab_set()

        _setup_dialog(self)
        if self._windowingsystem == 'aqua':
            self.bind("<Command-.>", self.cancel)
        self.bind('<Key-Escape>', self.cancel)
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.bind('<Key-Return>', self.ok)
        self.bind("<KP_Enter>", self.ok)

        self.create_widgets()
        self.update_idletasks()  # Need here for winfo_reqwidth below.
        self.geometry(  # Center dialog over parent (or below htest box).
                "+%d+%d" % (
                    parent.winfo_rootx() +
                    (parent.winfo_width()/2 - self.winfo_reqwidth()/2),
                    parent.winfo_rooty() +
                    ((parent.winfo_height()/2 - self.winfo_reqheight()/2)
                    if not _htest else 150)
                ) )
        self.resizable(height=False, width=False)

        if not _utest:
            self.deiconify()  # Unhide now that geometry set.
            self.entry.focus_set()
            self.wait_window()

    def create_widgets(self, ok_text='OK'):  # Do not replace.
        """Create entry (rows, extras, buttons.

        Entry stuff on rows 0-2, spanning cols 0-2.
        Buttons on row 99, cols 1, 2.
        """
        # Bind to self the widgets needed for entry_ok or unittest.
        self.frame = frame = Frame(self, padding=10)
        frame.grid(column=0, row=0, sticky='news')
        frame.grid_columnconfigure(0, weight=1)

        entrylabel = Label(frame, anchor='w', justify='left',
                           text=self.message)
        self.entryvar = StringVar(self, self.text0)
        self.entry = Entry(frame, width=30, textvariable=self.entryvar)
        self.error_font = Font(name='TkCaptionFont',
                               exists=True, root=self.parent)
        self.entry_error = Label(frame, text=' ', foreground='red',
                                 font=self.error_font)
        # Display or blank error by setting ['text'] =.
        entrylabel.grid(column=0, row=0, columnspan=3, padx=5, sticky=W)
        self.entry.grid(column=0, row=1, columnspan=3, padx=5, sticky=W+E,
                        pady=[10,0])
        self.entry_error.grid(column=0, row=2, columnspan=3, padx=5,
                              sticky=W+E)

        self.create_extra()

        self.button_ok = Button(
                frame, text=ok_text, default='active', command=self.ok)
        self.button_cancel = Button(
                frame, text='Cancel', command=self.cancel)

        self.button_ok.grid(column=1, row=99, padx=5)
        self.button_cancel.grid(column=2, row=99, padx=5)

    def create_extra(self): pass  # Override to add widgets.

    def showerror(self, message, widget=None):
        #self.bell(displayof=self)
        (widget or self.entry_error)['text'] = 'ERROR: ' + message

    def entry_ok(self):  # Example: usually replace.
        "Return non-blank entry or None."
        entry = self.entry.get().strip()
        if not entry:
            self.showerror('blank line.')
            return None
        return entry

    def ok(self, event=None):  # Do not replace.
        '''If entry is valid, bind it to 'result' and destroy tk widget.

        Otherwise leave dialog open for user to correct entry or cancel.
        '''
        self.entry_error['text'] = ''
        entry = self.entry_ok()
        if entry is not None:
            self.result = entry
            self.destroy()
        else:
            # [Ok] moves focus.  (<Return> does not.)  Move it back.
            self.entry.focus_set()

    def cancel(self, event=None):  # Do not replace.
        "Set dialog result to None and destroy tk widget."
        self.result = None
        self.destroy()

    def destroy(self):
        self.grab_release()
        super().destroy()


class SectionName(Query):
    "Get a name for a config file section name."
    # Used in ConfigDialog.GetNewKeysName, .GetNewThemeName (837)

    def __init__(self, parent, title, message, used_names,
                 *, _htest=False, _utest=False):
        super().__init__(parent, title, message, used_names=used_names,
                         _htest=_htest, _utest=_utest)

    def entry_ok(self):
        "Return sensible ConfigParser section name or None."
        name = self.entry.get().strip()
        if not name:
            self.showerror('no name specified.')
            return None
        elif len(name)>30:
            self.showerror('name is longer than 30 characters.')
            return None
        elif name in self.used_names:
            self.showerror('name is already in use.')
            return None
        return name


class ModuleName(Query):
    "Get a module name for Open Module menu entry."
    # Used in open_module (editor.EditorWindow until move to iobinding).

    def __init__(self, parent, title, message, text0,
                 *, _htest=False, _utest=False):
        super().__init__(parent, title, message, text0=text0,
                       _htest=_htest, _utest=_utest)

    def entry_ok(self):
        "Return entered module name as file path or None."
        name = self.entry.get().strip()
        if not name:
            self.showerror('no name specified.')
            return None
        # XXX Ought to insert current file's directory in front of path.
        try:
            spec = importlib.util.find_spec(name)
        except (ValueError, ImportError) as msg:
            self.showerror(str(msg))
            return None
        if spec is None:
            self.showerror("module not found.")
            return None
        if not isinstance(spec.loader, importlib.abc.SourceLoader):
            self.showerror("not a source-based module.")
            return None
        try:
            file_path = spec.loader.get_filename(name)
        except AttributeError:
            self.showerror("loader does not support get_filename.")
            return None
        except ImportError:
            # Some special modules require this (e.g. os.path)
            try:
                file_path = spec.loader.get_filename()
            except TypeError:
                self.showerror("loader failed to get filename.")
                return None
        return file_path


class Goto(Query):
    "Get a positive line number for editor Go To Line."
    # Used in editor.EditorWindow.goto_line_event.

    def entry_ok(self):
        try:
            lineno = int(self.entry.get())
        except ValueError:
            self.showerror('not a base 10 integer.')
            return None
        if lineno <= 0:
            self.showerror('not a positive integer.')
            return None
        return lineno


class HelpSource(Query):
    "Get menu name and help source for Help menu."
    # Used in ConfigDialog.HelpListItemAdd/Edit, (941/9)

    def __init__(self, parent, title, *, menuitem='', filepath='',
                 used_names={}, _htest=False, _utest=False):
        """Get menu entry and url/local file for Additional Help.

        User enters a name for the Help resource and a web url or file
        name. The user can browse for the file.
        """
        self.filepath = filepath
        message = 'Name for item on Help menu:'
        super().__init__(
                parent, title, message, text0=menuitem,
                used_names=used_names, _htest=_htest, _utest=_utest)

    def create_extra(self):
        "Add path widjets to rows 10-12."
        frame = self.frame
        pathlabel = Label(frame, anchor='w', justify='left',
                          text='Help File Path: Enter URL or browse for file')
        self.pathvar = StringVar(self, self.filepath)
        self.path = Entry(frame, textvariable=self.pathvar, width=40)
        browse = Button(frame, text='Browse', width=8,
                        command=self.browse_file)
        self.path_error = Label(frame, text=' ', foreground='red',
                                font=self.error_font)

        pathlabel.grid(column=0, row=10, columnspan=3, padx=5, pady=[10,0],
                       sticky=W)
        self.path.grid(column=0, row=11, columnspan=2, padx=5, sticky=W+E,
                       pady=[10,0])
        browse.grid(column=2, row=11, padx=5, sticky=W+S)
        self.path_error.grid(column=0, row=12, columnspan=3, padx=5,
                             sticky=W+E)

    def askfilename(self, filetypes, initdir, initfile):  # htest #
        # Extracted from browse_file so can mock for unittests.
        # Cannot unittest as cannot simulate button clicks.
        # Test by running htest, such as by running this file.
        return filedialog.Open(parent=self, filetypes=filetypes)\
               .show(initialdir=initdir, initialfile=initfile)

    def browse_file(self):
        filetypes = [
            ("HTML Files", "*.htm *.html", "TEXT"),
            ("PDF Files", "*.pdf", "TEXT"),
            ("Windows Help Files", "*.chm"),
            ("Text Files", "*.txt", "TEXT"),
            ("All Files", "*")]
        path = self.pathvar.get()
        if path:
            dir, base = os.path.split(path)
        else:
            base = None
            if platform[:3] == 'win':
                dir = os.path.join(os.path.dirname(executable), 'Doc')
                if not os.path.isdir(dir):
                    dir = os.getcwd()
            else:
                dir = os.getcwd()
        file = self.askfilename(filetypes, dir, base)
        if file:
            self.pathvar.set(file)

    item_ok = SectionName.entry_ok  # localize for test override

    def path_ok(self):
        "Simple validity check for menu file path"
        path = self.path.get().strip()
        if not path: #no path specified
            self.showerror('no help file path specified.', self.path_error)
            return None
        elif not path.startswith(('www.', 'http')):
            if path[:5] == 'file:':
                path = path[5:]
            if not os.path.exists(path):
                self.showerror('help file path does not exist.',
                               self.path_error)
                return None
            if platform == 'darwin':  # for Mac Safari
                path =  "file://" + path
        return path

    def entry_ok(self):
        "Return apparently valid (name, path) or None"
        self.path_error['text'] = ''
        name = self.item_ok()
        path = self.path_ok()
        return None if name is None or path is None else (name, path)

class CustomRun(Query):
    """Get settings for custom run of module.

    1. Command line arguments to extend sys.argv.
    2. Whether to restart Shell or not.
    """
    # Used in runscript.run_custom_event

    def __init__(self, parent, title, *, cli_args=[],
                 _htest=False, _utest=False):
        """cli_args is a list of strings.

        The list is assigned to the default Entry StringVar.
        The strings are displayed joined by ' ' for display.
        """
        message = 'Command Line Arguments for sys.argv:'
        super().__init__(
                parent, title, message, text0=cli_args,
                _htest=_htest, _utest=_utest)

    def create_extra(self):
        "Add run mode on rows 10-12."
        frame = self.frame
        self.restartvar = BooleanVar(self, value=True)
        restart = Checkbutton(frame, variable=self.restartvar, onvalue=True,
                              offvalue=False, text='Restart shell')
        self.args_error = Label(frame, text=' ', foreground='red',
                                font=self.error_font)

        restart.grid(column=0, row=10, columnspan=3, padx=5, sticky='w')
        self.args_error.grid(column=0, row=12, columnspan=3, padx=5,
                             sticky='we')

    def cli_args_ok(self):
        "Return command line arg list or None if error."
        cli_string = self.entry.get().strip()
        try:
            cli_args = shlex.split(cli_string, posix=True)
        except ValueError as err:
            self.showerror(str(err))
            return None
        return cli_args

    def entry_ok(self):
        "Return apparently valid (cli_args, restart) or None."
        cli_args = self.cli_args_ok()
        restart = self.restartvar.get()
        return None if cli_args is None else (cli_args, restart)


if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_query', verbosity=2, exit=False)

    from idlelib.idle_test.htest import run
    run(Query, HelpSource, CustomRun)
