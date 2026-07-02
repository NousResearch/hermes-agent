"""Execute code from an editor.

Check module: do a full syntax check of the current module.
Also run the tabnanny to catch any inconsistent tabs.

Run module: also execute the module's code in the __main__ namespace.
The window must have been saved previously. The module is added to
sys.modules, and is also added to the __main__ namespace.

TODO: Specify command line arguments in a dialog box.
"""
import os
import tabnanny
import time
import tokenize

from tkinter import messagebox

from idlelib.config import idleConf
from idlelib import macosx
from idlelib import pyshell
from idlelib.query import CustomRun
from idlelib import outwin

indent_message = """Error: Inconsistent indentation detected!

1) Your indentation is outright incorrect (easy to fix), OR

2) Your indentation mixes tabs and spaces.

To fix case 2, change all tabs to spaces by using Edit->Select All followed \
by Format->Untabify Region and specify the number of columns used by each tab.
"""


class ScriptBinding:

    def __init__(self, editwin):
        self.editwin = editwin
        # Provide instance variables referenced by debugger
        # XXX This should be done differently
        self.flist = self.editwin.flist
        self.root = self.editwin.root
        # cli_args is list of strings that extends sys.argv
        self.cli_args = []
        self.perf = 0.0    # Workaround for macOS 11 Uni2; see bpo-42508.

    def check_module_event(self, event):
        if isinstance(self.editwin, outwin.OutputWindow):
            self.editwin.text.bell()
            return 'break'
        filename = self.getfilename()
        if not filename:
            return 'break'
        if not self.checksyntax(filename):
            return 'break'
        if not self.tabnanny(filename):
            return 'break'
        return "break"

    def tabnanny(self, filename):
        # XXX: tabnanny should work on binary files as well
        with tokenize.open(filename) as f:
            try:
                tabnanny.process_tokens(tokenize.generate_tokens(f.readline))
            except tokenize.TokenError as msg:
                msgtxt, (lineno, start) = msg.args
                self.editwin.gotoline(lineno)
                self.errorbox("Tabnanny Tokenizing Error",
                              "Token Error: %s" % msgtxt)
                return False
            except tabnanny.NannyNag as nag:
                # The error messages from tabnanny are too confusing...
                self.editwin.gotoline(nag.get_lineno())
                self.errorbox("Tab/space error", indent_message)
                return False
        return True

    def checksyntax(self, filename):
        self.shell = shell = self.flist.open_shell()
        saved_stream = shell.get_warning_stream()
        shell.set_warning_stream(shell.stderr)
        with open(filename, 'rb') as f:
            source = f.read()
        if b'\r' in source:
            source = source.replace(b'\r\n', b'\n')
            source = source.replace(b'\r', b'\n')
        if source and source[-1] != ord(b'\n'):
            source = source + b'\n'
        editwin = self.editwin
        text = editwin.text
        text.tag_remove("ERROR", "1.0", "end")
        try:
            # If successful, return the compiled code
            return compile(source, filename, "exec")
        except (SyntaxError, OverflowError, ValueError) as value:
            msg = getattr(value, 'msg', '') or value or "<no detail available>"
            lineno = getattr(value, 'lineno', '') or 1
            offset = getattr(value, 'offset', '') or 0
            if offset == 0:
                lineno += 1  #mark end of offending line
            pos = "0.0 + %d lines + %d chars" % (lineno-1, offset-1)
            editwin.colorize_syntax_error(text, pos)
            self.errorbox("SyntaxError", "%-20s" % msg)
            return False
        finally:
            shell.set_warning_stream(saved_stream)

    def run_custom_event(self, event):
        return self.run_module_event(event, customize=True)

    def run_module_event(self, event, *, customize=False):
        """Run the module after setting up the environment.

        First check the syntax.  Next get customization.  If OK, make
        sure the shell is active and then transfer the arguments, set
        the run environment's working directory to the directory of the
        module being executed and also add that directory to its
        sys.path if not already included.
        """
        if macosx.isCocoaTk() and (time.perf_counter() - self.perf < .05):
            return 'break'
        if isinstance(self.editwin, outwin.OutputWindow):
            self.editwin.text.bell()
            return 'break'
        filename = self.getfilename()
        if not filename:
            return 'break'
        code = self.checksyntax(filename)
        if not code:
            return 'break'
        if not self.tabnanny(filename):
            return 'break'
        if customize:
            title = f"Customize {self.editwin.short_title()} Run"
            run_args = CustomRun(self.shell.text, title,
                                 cli_args=self.cli_args).result
            if not run_args:  # User cancelled.
                return 'break'
        self.cli_args, restart = run_args if customize else ([], True)
        interp = self.shell.interp
        if pyshell.use_subprocess and restart:
            interp.restart_subprocess(
                    with_cwd=False, filename=filename)
        dirname = os.path.dirname(filename)
        argv = [filename]
        if self.cli_args:
            argv += self.cli_args
        interp.runcommand(f"""if 1:
            __file__ = {filename!r}
            import sys as _sys
            from os.path import basename as _basename
            argv = {argv!r}
            if (not _sys.argv or
                _basename(_sys.argv[0]) != _basename(__file__) or
                len(argv) > 1):
                _sys.argv = argv
            import os as _os
            _os.chdir({dirname!r})
            del _sys, argv, _basename, _os
            \n""")
        interp.prepend_syspath(filename)
        # XXX KBK 03Jul04 When run w/o subprocess, runtime warnings still
        #         go to __stderr__.  With subprocess, they go to the shell.
        #         Need to change streams in pyshell.ModifiedInterpreter.
        interp.runcode(code)
        return 'break'

    def getfilename(self):
        """Get source filename.  If not saved, offer to save (or create) file

        The debugger requires a source file.  Make sure there is one, and that
        the current version of the source buffer has been saved.  If the user
        declines to save or cancels the Save As dialog, return None.

        If the user has configured IDLE for Autosave, the file will be
        silently saved if it already exists and is dirty.

        """
        filename = self.editwin.io.filename
        if not self.editwin.get_saved():
            autosave = idleConf.GetOption('main', 'General',
                                          'autosave', type='bool')
            if autosave and filename:
                self.editwin.io.save(None)
            else:
                confirm = self.ask_save_dialog()
                self.editwin.text.focus_set()
                if confirm:
                    self.editwin.io.save(None)
                    filename = self.editwin.io.filename
                else:
                    filename = None
        return filename

    def ask_save_dialog(self):
        msg = "Source Must Be Saved\n" + 5*' ' + "OK to Save?"
        confirm = messagebox.askokcancel(title="Save Before Run or Check",
                                           message=msg,
                                           default=messagebox.OK,
                                           parent=self.editwin.text)
        return confirm

    def errorbox(self, title, message):
        # XXX This should really be a function of EditorWindow...
        messagebox.showerror(title, message, parent=self.editwin.text)
        self.editwin.text.focus_set()
        self.perf = time.perf_counter()


if __name__ == "__main__":
    from unittest import main
    main('idlelib.idle_test.test_runscript', verbosity=2,)
