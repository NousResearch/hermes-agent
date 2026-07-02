#! /usr/bin/env python3

import sys
if __name__ == "__main__":
    sys.modules['idlelib.pyshell'] = sys.modules['__main__']

try:
    from tkinter import *
except ImportError:
    print("** IDLE can't import Tkinter.\n"
          "Your Python may not be configured for Tk. **", file=sys.__stderr__)
    raise SystemExit(1)

# Valid arguments for the ...Awareness call below are defined in the following.
# https://msdn.microsoft.com/en-us/library/windows/desktop/dn280512(v=vs.85).aspx
if sys.platform == 'win32':
    try:
        import ctypes
        PROCESS_SYSTEM_DPI_AWARE = 1  # Int required.
        ctypes.OleDLL('shcore').SetProcessDpiAwareness(PROCESS_SYSTEM_DPI_AWARE)
    except (ImportError, AttributeError, OSError):
        pass

from tkinter import messagebox

from code import InteractiveInterpreter
import itertools
import linecache
import os
import os.path
from platform import python_version
import re
import socket
import subprocess
from textwrap import TextWrapper
import threading
import time
import tokenize
import warnings

from idlelib.colorizer import ColorDelegator
from idlelib.config import idleConf
from idlelib.delegator import Delegator
from idlelib import debugger
from idlelib import debugger_r
from idlelib.editor import EditorWindow, fixwordbreaks
from idlelib.filelist import FileList
from idlelib.outwin import OutputWindow
from idlelib import replace
from idlelib import rpc
from idlelib.run import idle_formatwarning, StdInputFile, StdOutputFile
from idlelib.undo import UndoDelegator

# Default for testing; defaults to True in main() for running.
use_subprocess = False

HOST = '127.0.0.1' # python execution server on localhost loopback
PORT = 0  # someday pass in host, port for remote debug capability

try:  # In case IDLE started with -n.
    eof = 'Ctrl-D (end-of-file)'
    exit.eof = eof
    quit.eof = eof
except NameError: # In case python started with -S.
    pass

# Override warnings module to write to warning_stream.  Initialize to send IDLE
# internal warnings to the console.  ScriptBinding.check_syntax() will
# temporarily redirect the stream to the shell window to display warnings when
# checking user's code.
warning_stream = sys.__stderr__  # None, at least on Windows, if no console.

def idle_showwarning(
        message, category, filename, lineno, file=None, line=None):
    """Show Idle-format warning (after replacing warnings.showwarning).

    The differences are the formatter called, the file=None replacement,
    which can be None, the capture of the consequence AttributeError,
    and the output of a hard-coded prompt.
    """
    if file is None:
        file = warning_stream
    try:
        file.write(idle_formatwarning(
                message, category, filename, lineno, line=line))
        file.write(">>> ")
    except (AttributeError, OSError):
        pass  # if file (probably __stderr__) is invalid, skip warning.

_warnings_showwarning = None

def capture_warnings(capture):
    "Replace warning.showwarning with idle_showwarning, or reverse."

    global _warnings_showwarning
    if capture:
        if _warnings_showwarning is None:
            _warnings_showwarning = warnings.showwarning
            warnings.showwarning = idle_showwarning
    else:
        if _warnings_showwarning is not None:
            warnings.showwarning = _warnings_showwarning
            _warnings_showwarning = None

capture_warnings(True)

def extended_linecache_checkcache(filename=None,
                                  orig_checkcache=linecache.checkcache):
    """Extend linecache.checkcache to preserve the <pyshell#...> entries

    Rather than repeating the linecache code, patch it to save the
    <pyshell#...> entries, call the original linecache.checkcache()
    (skipping them), and then restore the saved entries.

    orig_checkcache is bound at definition time to the original
    method, allowing it to be patched.
    """
    cache = linecache.cache
    save = {}
    for key in list(cache):
        if key[:1] + key[-1:] == '<>':
            save[key] = cache.pop(key)
    orig_checkcache(filename)
    cache.update(save)

# Patch linecache.checkcache():
linecache.checkcache = extended_linecache_checkcache


class PyShellEditorWindow(EditorWindow):
    "Regular text edit window in IDLE, supports breakpoints"

    def __init__(self, *args):
        self.breakpoints = []
        EditorWindow.__init__(self, *args)
        self.text.bind("<<set-breakpoint>>", self.set_breakpoint_event)
        self.text.bind("<<clear-breakpoint>>", self.clear_breakpoint_event)
        self.text.bind("<<open-python-shell>>", self.flist.open_shell)

        #TODO: don't read/write this from/to .idlerc when testing
        self.breakpointPath = os.path.join(
                idleConf.userdir, 'breakpoints.lst')
        # whenever a file is changed, restore breakpoints
        def filename_changed_hook(old_hook=self.io.filename_change_hook,
                                  self=self):
            self.restore_file_breaks()
            old_hook()
        self.io.set_filename_change_hook(filename_changed_hook)
        if self.io.filename:
            self.restore_file_breaks()
        self.color_breakpoint_text()

    rmenu_specs = [
        ("Cut", "<<cut>>", "rmenu_check_cut"),
        ("Copy", "<<copy>>", "rmenu_check_copy"),
        ("Paste", "<<paste>>", "rmenu_check_paste"),
        (None, None, None),
        ("Set Breakpoint", "<<set-breakpoint>>", None),
        ("Clear Breakpoint", "<<clear-breakpoint>>", None)
    ]

    def color_breakpoint_text(self, color=True):
        "Turn colorizing of breakpoint text on or off"
        if self.io is None:
            # possible due to update in restore_file_breaks
            return
        if color:
            theme = idleConf.CurrentTheme()
            cfg = idleConf.GetHighlight(theme, "break")
        else:
            cfg = {'foreground': '', 'background': ''}
        self.text.tag_config('BREAK', cfg)

    def set_breakpoint(self, lineno):
        text = self.text
        filename = self.io.filename
        text.tag_add("BREAK", "%d.0" % lineno, "%d.0" % (lineno+1))
        try:
            self.breakpoints.index(lineno)
        except ValueError:  # only add if missing, i.e. do once
            self.breakpoints.append(lineno)
        try:    # update the subprocess debugger
            debug = self.flist.pyshell.interp.debugger
            debug.set_breakpoint(filename, lineno)
        except: # but debugger may not be active right now....
            pass

    def set_breakpoint_event(self, event=None):
        text = self.text
        filename = self.io.filename
        if not filename:
            text.bell()
            return
        lineno = int(float(text.index("insert")))
        self.set_breakpoint(lineno)

    def clear_breakpoint_event(self, event=None):
        text = self.text
        filename = self.io.filename
        if not filename:
            text.bell()
            return
        lineno = int(float(text.index("insert")))
        try:
            self.breakpoints.remove(lineno)
        except:
            pass
        text.tag_remove("BREAK", "insert linestart",\
                        "insert lineend +1char")
        try:
            debug = self.flist.pyshell.interp.debugger
            debug.clear_breakpoint(filename, lineno)
        except:
            pass

    def clear_file_breaks(self):
        if self.breakpoints:
            text = self.text
            filename = self.io.filename
            if not filename:
                text.bell()
                return
            self.breakpoints = []
            text.tag_remove("BREAK", "1.0", END)
            try:
                debug = self.flist.pyshell.interp.debugger
                debug.clear_file_breaks(filename)
            except:
                pass

    def store_file_breaks(self):
        "Save breakpoints when file is saved"
        # XXX 13 Dec 2002 KBK Currently the file must be saved before it can
        #     be run.  The breaks are saved at that time.  If we introduce
        #     a temporary file save feature the save breaks functionality
        #     needs to be re-verified, since the breaks at the time the
        #     temp file is created may differ from the breaks at the last
        #     permanent save of the file.  Currently, a break introduced
        #     after a save will be effective, but not persistent.
        #     This is necessary to keep the saved breaks synched with the
        #     saved file.
        #
        #     Breakpoints are set as tagged ranges in the text.
        #     Since a modified file has to be saved before it is
        #     run, and since self.breakpoints (from which the subprocess
        #     debugger is loaded) is updated during the save, the visible
        #     breaks stay synched with the subprocess even if one of these
        #     unexpected breakpoint deletions occurs.
        breaks = self.breakpoints
        filename = self.io.filename
        try:
            with open(self.breakpointPath) as fp:
                lines = fp.readlines()
        except OSError:
            lines = []
        try:
            with open(self.breakpointPath, "w") as new_file:
                for line in lines:
                    if not line.startswith(filename + '='):
                        new_file.write(line)
                self.update_breakpoints()
                breaks = self.breakpoints
                if breaks:
                    new_file.write(filename + '=' + str(breaks) + '\n')
        except OSError as err:
            if not getattr(self.root, "breakpoint_error_displayed", False):
                self.root.breakpoint_error_displayed = True
                messagebox.showerror(title='IDLE Error',
                    message='Unable to update breakpoint list:\n%s'
                        % str(err),
                    parent=self.text)

    def restore_file_breaks(self):
        self.text.update()   # this enables setting "BREAK" tags to be visible
        if self.io is None:
            # can happen if IDLE closes due to the .update() call
            return
        filename = self.io.filename
        if filename is None:
            return
        if os.path.isfile(self.breakpointPath):
            with open(self.breakpointPath) as fp:
                lines = fp.readlines()
            for line in lines:
                if line.startswith(filename + '='):
                    breakpoint_linenumbers = eval(line[len(filename)+1:])
                    for breakpoint_linenumber in breakpoint_linenumbers:
                        self.set_breakpoint(breakpoint_linenumber)

    def update_breakpoints(self):
        "Retrieves all the breakpoints in the current window"
        text = self.text
        ranges = text.tag_ranges("BREAK")
        linenumber_list = self.ranges_to_linenumbers(ranges)
        self.breakpoints = linenumber_list

    def ranges_to_linenumbers(self, ranges):
        lines = []
        for index in range(0, len(ranges), 2):
            lineno = int(float(ranges[index].string))
            end = int(float(ranges[index+1].string))
            while lineno < end:
                lines.append(lineno)
                lineno += 1
        return lines

# XXX 13 Dec 2002 KBK Not used currently
#    def saved_change_hook(self):
#        "Extend base method - clear breaks if module is modified"
#        if not self.get_saved():
#            self.clear_file_breaks()
#        EditorWindow.saved_change_hook(self)

    def _close(self):
        "Extend base method - clear breaks when module is closed"
        self.clear_file_breaks()
        EditorWindow._close(self)


class PyShellFileList(FileList):
    "Extend base class: IDLE supports a shell and breakpoints"

    # override FileList's class variable, instances return PyShellEditorWindow
    # instead of EditorWindow when new edit windows are created.
    EditorWindow = PyShellEditorWindow

    pyshell = None

    def open_shell(self, event=None):
        if self.pyshell:
            self.pyshell.top.wakeup()
        else:
            self.pyshell = PyShell(self)
            if self.pyshell:
                if not self.pyshell.begin():
                    return None
        return self.pyshell


class ModifiedColorDelegator(ColorDelegator):
    "Extend base class: colorizer for the shell window itself"
    def recolorize_main(self):
        self.tag_remove("TODO", "1.0", "iomark")
        self.tag_add("SYNC", "1.0", "iomark")
        ColorDelegator.recolorize_main(self)

    def removecolors(self):
        # Don't remove shell color tags before "iomark"
        for tag in self.tagdefs:
            self.tag_remove(tag, "iomark", "end")


class ModifiedUndoDelegator(UndoDelegator):
    "Extend base class: forbid insert/delete before the I/O mark"
    def insert(self, index, chars, tags=None):
        try:
            if self.delegate.compare(index, "<", "iomark"):
                self.delegate.bell()
                return
        except TclError:
            pass
        UndoDelegator.insert(self, index, chars, tags)

    def delete(self, index1, index2=None):
        try:
            if self.delegate.compare(index1, "<", "iomark"):
                self.delegate.bell()
                return
        except TclError:
            pass
        UndoDelegator.delete(self, index1, index2)

    def undo_event(self, event):
        # Temporarily monkey-patch the delegate's .insert() method to
        # always use the "stdin" tag.  This is needed for undo-ing
        # deletions to preserve the "stdin" tag, because UndoDelegator
        # doesn't preserve tags for deleted text.
        orig_insert = self.delegate.insert
        self.delegate.insert = \
            lambda index, chars: orig_insert(index, chars, "stdin")
        try:
            super().undo_event(event)
        finally:
            self.delegate.insert = orig_insert


class UserInputTaggingDelegator(Delegator):
    """Delegator used to tag user input with "stdin"."""
    def insert(self, index, chars, tags=None):
        if tags is None:
            tags = "stdin"
        self.delegate.insert(index, chars, tags)


class MyRPCClient(rpc.RPCClient):

    def handle_EOF(self):
        "Override the base class - just re-raise EOFError"
        raise EOFError

def restart_line(width, filename):  # See bpo-38141.
    """Return width long restart line formatted with filename.

    Fill line with balanced '='s, with any extras and at least one at
    the beginning.  Do not end with a trailing space.
    """
    tag = f"= RESTART: {filename or 'Shell'} ="
    if width >= len(tag):
        div, mod = divmod((width -len(tag)), 2)
        return f"{(div+mod)*'='}{tag}{div*'='}"
    else:
        return tag[:-2]  # Remove ' ='.


class ModifiedInterpreter(InteractiveInterpreter):

    def __init__(self, tkconsole):
        self.tkconsole = tkconsole
        locals = sys.modules['__main__'].__dict__
        InteractiveInterpreter.__init__(self, locals=locals)
        self.restarting = False
        self.subprocess_arglist = None
        self.port = PORT
        self.original_compiler_flags = self.compile.compiler.flags

    _afterid = None
    rpcclt = None
    rpcsubproc = None

    def spawn_subprocess(self):
        if self.subprocess_arglist is None:
            self.subprocess_arglist = self.build_subprocess_arglist()
        self.rpcsubproc = subprocess.Popen(self.subprocess_arglist)

    def build_subprocess_arglist(self):
        assert (self.port!=0), (
            "Socket should have been assigned a port number.")
        w = ['-W' + s for s in sys.warnoptions]
        # Maybe IDLE is installed and is being accessed via sys.path,
        # or maybe it's not installed and the idle.py script is being
        # run from the IDLE source directory.
        del_exitf = idleConf.GetOption('main', 'General', 'delete-exitfunc',
                                       default=False, type='bool')
        command = f"__import__('idlelib.run').run.main({del_exitf!r})"
        return [sys.executable] + w + ["-c", command, str(self.port)]

    def start_subprocess(self):
        addr = (HOST, self.port)
        # GUI makes several attempts to acquire socket, listens for connection
        for i in range(3):
            time.sleep(i)
            try:
                self.rpcclt = MyRPCClient(addr)
                break
            except OSError:
                pass
        else:
            self.display_port_binding_error()
            return None
        # if PORT was 0, system will assign an 'ephemeral' port. Find it out:
        self.port = self.rpcclt.listening_sock.getsockname()[1]
        # if PORT was not 0, probably working with a remote execution server
        if PORT != 0:
            # To allow reconnection within the 2MSL wait (cf. Stevens TCP
            # V1, 18.6),  set SO_REUSEADDR.  Note that this can be problematic
            # on Windows since the implementation allows two active sockets on
            # the same address!
            self.rpcclt.listening_sock.setsockopt(socket.SOL_SOCKET,
                                           socket.SO_REUSEADDR, 1)
        self.spawn_subprocess()
        #time.sleep(20) # test to simulate GUI not accepting connection
        # Accept the connection from the Python execution server
        self.rpcclt.listening_sock.settimeout(10)
        try:
            self.rpcclt.accept()
        except TimeoutError:
            self.display_no_subprocess_error()
            return None
        self.rpcclt.register("console", self.tkconsole)
        self.rpcclt.register("stdin", self.tkconsole.stdin)
        self.rpcclt.register("stdout", self.tkconsole.stdout)
        self.rpcclt.register("stderr", self.tkconsole.stderr)
        self.rpcclt.register("flist", self.tkconsole.flist)
        self.rpcclt.register("linecache", linecache)
        self.rpcclt.register("interp", self)
        self.transfer_path(with_cwd=True)
        self.poll_subprocess()
        return self.rpcclt

    def restart_subprocess(self, with_cwd=False, filename=''):
        if self.restarting:
            return self.rpcclt
        self.restarting = True
        # close only the subprocess debugger
        debug = self.getdebugger()
        if debug:
            try:
                # Only close subprocess debugger, don't unregister gui_adap!
                debugger_r.close_subprocess_debugger(self.rpcclt)
            except:
                pass
        # Kill subprocess, spawn a new one, accept connection.
        self.rpcclt.close()
        self.terminate_subprocess()
        console = self.tkconsole
        was_executing = console.executing
        console.executing = False
        self.spawn_subprocess()
        try:
            self.rpcclt.accept()
        except TimeoutError:
            self.display_no_subprocess_error()
            return None
        self.transfer_path(with_cwd=with_cwd)
        console.stop_readline()
        # annotate restart in shell window and mark it
        console.text.delete("iomark", "end-1c")
        console.write('\n')
        console.write(restart_line(console.width, filename))
        console.text.mark_set("restart", "end-1c")
        console.text.mark_gravity("restart", "left")
        if not filename:
            console.showprompt()
        # restart subprocess debugger
        if debug:
            # Restarted debugger connects to current instance of debug GUI
            debugger_r.restart_subprocess_debugger(self.rpcclt)
            # reload remote debugger breakpoints for all PyShellEditWindows
            debug.load_breakpoints()
        self.compile.compiler.flags = self.original_compiler_flags
        self.restarting = False
        return self.rpcclt

    def __request_interrupt(self):
        self.rpcclt.remotecall("exec", "interrupt_the_server", (), {})

    def interrupt_subprocess(self):
        threading.Thread(target=self.__request_interrupt).start()

    def kill_subprocess(self):
        if self._afterid is not None:
            self.tkconsole.text.after_cancel(self._afterid)
        try:
            self.rpcclt.listening_sock.close()
        except AttributeError:  # no socket
            pass
        try:
            self.rpcclt.close()
        except AttributeError:  # no socket
            pass
        self.terminate_subprocess()
        self.tkconsole.executing = False
        self.rpcclt = None

    def terminate_subprocess(self):
        "Make sure subprocess is terminated"
        try:
            self.rpcsubproc.kill()
        except OSError:
            # process already terminated
            return
        else:
            try:
                self.rpcsubproc.wait()
            except OSError:
                return

    def transfer_path(self, with_cwd=False):
        if with_cwd:        # Issue 13506
            path = ['']     # include Current Working Directory
            path.extend(sys.path)
        else:
            path = sys.path

        self.runcommand("""if 1:
        import sys as _sys
        _sys.path = {!r}
        del _sys
        \n""".format(path))

    active_seq = None

    def poll_subprocess(self):
        clt = self.rpcclt
        if clt is None:
            return
        try:
            response = clt.pollresponse(self.active_seq, wait=0.05)
        except (EOFError, OSError, KeyboardInterrupt):
            # lost connection or subprocess terminated itself, restart
            # [the KBI is from rpc.SocketIO.handle_EOF()]
            if self.tkconsole.closing:
                return
            response = None
            self.restart_subprocess()
        if response:
            self.tkconsole.resetoutput()
            self.active_seq = None
            how, what = response
            console = self.tkconsole.console
            if how == "OK":
                if what is not None:
                    print(repr(what), file=console)
            elif how == "EXCEPTION":
                if self.tkconsole.getvar("<<toggle-jit-stack-viewer>>"):
                    self.remote_stack_viewer()
            elif how == "ERROR":
                errmsg = "pyshell.ModifiedInterpreter: Subprocess ERROR:\n"
                print(errmsg, what, file=sys.__stderr__)
                print(errmsg, what, file=console)
            # we received a response to the currently active seq number:
            try:
                self.tkconsole.endexecuting()
            except AttributeError:  # shell may have closed
                pass
        # Reschedule myself
        if not self.tkconsole.closing:
            self._afterid = self.tkconsole.text.after(
                self.tkconsole.pollinterval, self.poll_subprocess)

    debugger = None

    def setdebugger(self, debugger):
        self.debugger = debugger

    def getdebugger(self):
        return self.debugger

    def open_remote_stack_viewer(self):
        """Initiate the remote stack viewer from a separate thread.

        This method is called from the subprocess, and by returning from this
        method we allow the subprocess to unblock.  After a bit the shell
        requests the subprocess to open the remote stack viewer which returns a
        static object looking at the last exception.  It is queried through
        the RPC mechanism.

        """
        self.tkconsole.text.after(300, self.remote_stack_viewer)
        return

    def remote_stack_viewer(self):
        from idlelib import debugobj_r
        oid = self.rpcclt.remotequeue("exec", "stackviewer", ("flist",), {})
        if oid is None:
            self.tkconsole.root.bell()
            return
        item = debugobj_r.StubObjectTreeItem(self.rpcclt, oid)
        from idlelib.tree import ScrolledCanvas, TreeNode
        top = Toplevel(self.tkconsole.root)
        theme = idleConf.CurrentTheme()
        background = idleConf.GetHighlight(theme, 'normal')['background']
        sc = ScrolledCanvas(top, bg=background, highlightthickness=0)
        sc.frame.pack(expand=1, fill="both")
        node = TreeNode(sc.canvas, None, item)
        node.expand()
        # XXX Should GC the remote tree when closing the window

    gid = 0

    def execsource(self, source):
        "Like runsource() but assumes complete exec source"
        filename = self.stuffsource(source)
        self.execfile(filename, source)

    def execfile(self, filename, source=None):
        "Execute an existing file"
        if source is None:
            with tokenize.open(filename) as fp:
                source = fp.read()
                if use_subprocess:
                    source = (f"__file__ = r'''{os.path.abspath(filename)}'''\n"
                              + source + "\ndel __file__")
        try:
            code = compile(source, filename, "exec")
        except (OverflowError, SyntaxError):
            self.tkconsole.resetoutput()
            print('*** Error in script or command!\n'
                 'Traceback (most recent call last):',
                  file=self.tkconsole.stderr)
            InteractiveInterpreter.showsyntaxerror(self, filename)
            self.tkconsole.showprompt()
        else:
            self.runcode(code)

    def runsource(self, source):
        "Extend base class method: Stuff the source in the line cache first"
        filename = self.stuffsource(source)
        # at the moment, InteractiveInterpreter expects str
        assert isinstance(source, str)
        # InteractiveInterpreter.runsource() calls its runcode() method,
        # which is overridden (see below)
        return InteractiveInterpreter.runsource(self, source, filename)

    def stuffsource(self, source):
        "Stuff source in the filename cache"
        filename = "<pyshell#%d>" % self.gid
        self.gid = self.gid + 1
        lines = source.split("\n")
        linecache.cache[filename] = len(source)+1, 0, lines, filename
        return filename

    def prepend_syspath(self, filename):
        "Prepend sys.path with file's directory if not already included"
        self.runcommand("""if 1:
            _filename = {!r}
            import sys as _sys
            from os.path import dirname as _dirname
            _dir = _dirname(_filename)
            if not _dir in _sys.path:
                _sys.path.insert(0, _dir)
            del _filename, _sys, _dirname, _dir
            \n""".format(filename))

    def showsyntaxerror(self, filename=None):
        """Override Interactive Interpreter method: Use Colorizing

        Color the offending position instead of printing it and pointing at it
        with a caret.

        """
        tkconsole = self.tkconsole
        text = tkconsole.text
        text.tag_remove("ERROR", "1.0", "end")
        type, value, tb = sys.exc_info()
        msg = getattr(value, 'msg', '') or value or "<no detail available>"
        lineno = getattr(value, 'lineno', '') or 1
        offset = getattr(value, 'offset', '') or 0
        if offset == 0:
            lineno += 1 #mark end of offending line
        if lineno == 1:
            pos = "iomark + %d chars" % (offset-1)
        else:
            pos = "iomark linestart + %d lines + %d chars" % \
                  (lineno-1, offset-1)
        tkconsole.colorize_syntax_error(text, pos)
        tkconsole.resetoutput()
        self.write("SyntaxError: %s\n" % msg)
        tkconsole.showprompt()

    def showtraceback(self):
        "Extend base class method to reset output properly"
        self.tkconsole.resetoutput()
        self.checklinecache()
        InteractiveInterpreter.showtraceback(self)
        if self.tkconsole.getvar("<<toggle-jit-stack-viewer>>"):
            self.tkconsole.open_stack_viewer()

    def checklinecache(self):
        "Remove keys other than '<pyshell#n>'."
        cache = linecache.cache
        for key in list(cache):  # Iterate list because mutate cache.
            if key[:1] + key[-1:] != "<>":
                del cache[key]

    def runcommand(self, code):
        "Run the code without invoking the debugger"
        # The code better not raise an exception!
        if self.tkconsole.executing:
            self.display_executing_dialog()
            return 0
        if self.rpcclt:
            self.rpcclt.remotequeue("exec", "runcode", (code,), {})
        else:
            exec(code, self.locals)
        return 1

    def runcode(self, code):
        "Override base class method"
        if self.tkconsole.executing:
            self.restart_subprocess()
        self.checklinecache()
        debugger = self.debugger
        try:
            self.tkconsole.beginexecuting()
            if not debugger and self.rpcclt is not None:
                self.active_seq = self.rpcclt.asyncqueue("exec", "runcode",
                                                        (code,), {})
            elif debugger:
                debugger.run(code, self.locals)
            else:
                exec(code, self.locals)
        except SystemExit:
            if not self.tkconsole.closing:
                if messagebox.askyesno(
                    "Exit?",
                    "Do you want to exit altogether?",
                    default="yes",
                    parent=self.tkconsole.text):
                    raise
                else:
                    self.showtraceback()
            else:
                raise
        except:
            if use_subprocess:
                print("IDLE internal error in runcode()",
                      file=self.tkconsole.stderr)
                self.showtraceback()
                self.tkconsole.endexecuting()
            else:
                if self.tkconsole.canceled:
                    self.tkconsole.canceled = False
                    print("KeyboardInterrupt", file=self.tkconsole.stderr)
                else:
                    self.showtraceback()
        finally:
            if not use_subprocess:
                try:
                    self.tkconsole.endexecuting()
                except AttributeError:  # shell may have closed
                    pass

    def write(self, s):
        "Override base class method"
        return self.tkconsole.stderr.write(s)

    def display_port_binding_error(self):
        messagebox.showerror(
            "Port Binding Error",
            "IDLE can't bind to a TCP/IP port, which is necessary to "
            "communicate with its Python execution server.  This might be "
            "because no networking is installed on this computer.  "
            "Run IDLE with the -n command line switch to start without a "
            "subprocess and refer to Help/IDLE Help 'Running without a "
            "subprocess' for further details.",
            parent=self.tkconsole.text)

    def display_no_subprocess_error(self):
        messagebox.showerror(
            "Subprocess Connection Error",
            "IDLE's subprocess didn't make connection.\n"
            "See the 'Startup failure' section of the IDLE doc, online at\n"
            "https://docs.python.org/3/library/idle.html#startup-failure",
            parent=self.tkconsole.text)

    def display_executing_dialog(self):
        messagebox.showerror(
            "Already executing",
            "The Python Shell window is already executing a command; "
            "please wait until it is finished.",
            parent=self.tkconsole.text)


class PyShell(OutputWindow):
    from idlelib.squeezer import Squeezer

    shell_title = "IDLE Shell " + python_version()

    # Override classes
    ColorDelegator = ModifiedColorDelegator
    UndoDelegator = ModifiedUndoDelegator

    # Override menus
    menu_specs = [
        ("file", "_File"),
        ("edit", "_Edit"),
        ("debug", "_Debug"),
        ("options", "_Options"),
        ("window", "_Window"),
        ("help", "_Help"),
    ]

    # Extend right-click context menu
    rmenu_specs = OutputWindow.rmenu_specs + [
        ("Squeeze", "<<squeeze-current-text>>"),
    ]
    _idx = 1 + len(list(itertools.takewhile(
        lambda rmenu_item: rmenu_item[0] != "Copy", rmenu_specs)
    ))
    rmenu_specs.insert(_idx, ("Copy with prompts",
                              "<<copy-with-prompts>>",
                              "rmenu_check_copy"))
    del _idx

    allow_line_numbers = False
    user_input_insert_tags = "stdin"

    # New classes
    from idlelib.history import History
    from idlelib.sidebar import ShellSidebar

    def __init__(self, flist=None):
        if use_subprocess:
            ms = self.menu_specs
            if ms[2][0] != "shell":
                ms.insert(2, ("shell", "She_ll"))
        self.interp = ModifiedInterpreter(self)
        if flist is None:
            root = Tk()
            fixwordbreaks(root)
            root.withdraw()
            flist = PyShellFileList(root)

        self.shell_sidebar = None  # initialized below

        OutputWindow.__init__(self, flist, None, None)

        self.usetabs = False
        # indentwidth must be 8 when using tabs.  See note in EditorWindow:
        self.indentwidth = 4

        self.sys_ps1 = sys.ps1 if hasattr(sys, 'ps1') else '>>>\n'
        self.prompt_last_line = self.sys_ps1.split('\n')[-1]
        self.prompt = self.sys_ps1  # Changes when debug active

        text = self.text
        text.configure(wrap="char")
        text.bind("<<newline-and-indent>>", self.enter_callback)
        text.bind("<<plain-newline-and-indent>>", self.linefeed_callback)
        text.bind("<<interrupt-execution>>", self.cancel_callback)
        text.bind("<<end-of-file>>", self.eof_callback)
        text.bind("<<open-stack-viewer>>", self.open_stack_viewer)
        text.bind("<<toggle-debugger>>", self.toggle_debugger)
        text.bind("<<toggle-jit-stack-viewer>>", self.toggle_jit_stack_viewer)
        text.bind("<<copy-with-prompts>>", self.copy_with_prompts_callback)
        if use_subprocess:
            text.bind("<<view-restart>>", self.view_restart_mark)
            text.bind("<<restart-shell>>", self.restart_shell)
        self.squeezer = self.Squeezer(self)
        text.bind("<<squeeze-current-text>>",
                  self.squeeze_current_text_event)

        self.save_stdout = sys.stdout
        self.save_stderr = sys.stderr
        self.save_stdin = sys.stdin
        from idlelib import iomenu
        self.stdin = StdInputFile(self, "stdin",
                                  iomenu.encoding, iomenu.errors)
        self.stdout = StdOutputFile(self, "stdout",
                                    iomenu.encoding, iomenu.errors)
        self.stderr = StdOutputFile(self, "stderr",
                                    iomenu.encoding, "backslashreplace")
        self.console = StdOutputFile(self, "console",
                                     iomenu.encoding, iomenu.errors)
        if not use_subprocess:
            sys.stdout = self.stdout
            sys.stderr = self.stderr
            sys.stdin = self.stdin
        try:
            # page help() text to shell.
            import pydoc # import must be done here to capture i/o rebinding.
            # XXX KBK 27Dec07 use text viewer someday, but must work w/o subproc
            pydoc.pager = pydoc.plainpager
        except:
            sys.stderr = sys.__stderr__
            raise
        #
        self.history = self.History(self.text)
        #
        self.pollinterval = 50  # millisec

        self.shell_sidebar = self.ShellSidebar(self)

        # Insert UserInputTaggingDelegator at the top of the percolator,
        # but make calls to text.insert() skip it.  This causes only insert
        # events generated in Tcl/Tk to go through this delegator.
        self.text.insert = self.per.top.insert
        self.per.insertfilter(UserInputTaggingDelegator())

    def ResetFont(self):
        super().ResetFont()

        if self.shell_sidebar is not None:
            self.shell_sidebar.update_font()

    def ResetColorizer(self):
        super().ResetColorizer()

        theme = idleConf.CurrentTheme()
        tag_colors = {
          "stdin": {'background': None, 'foreground': None},
          "stdout": idleConf.GetHighlight(theme, "stdout"),
          "stderr": idleConf.GetHighlight(theme, "stderr"),
          "console": idleConf.GetHighlight(theme, "normal"),
        }
        for tag, tag_colors_config in tag_colors.items():
            self.text.tag_configure(tag, **tag_colors_config)

        if self.shell_sidebar is not None:
            self.shell_sidebar.update_colors()

    def replace_event(self, event):
        replace.replace(self.text, insert_tags="stdin")
        return "break"

    def get_standard_extension_names(self):
        return idleConf.GetExtensions(shell_only=True)

    def get_prompt_text(self, first, last):
        """Return text between first and last with prompts added."""
        text = self.text.get(first, last)
        lineno_range = range(
            int(float(first)),
            int(float(last))
         )
        prompts = [
            self.shell_sidebar.line_prompts.get(lineno)
            for lineno in lineno_range
        ]
        return "\n".join(
            line if prompt is None else f"{prompt} {line}"
            for prompt, line in zip(prompts, text.splitlines())
        ) + "\n"


    def copy_with_prompts_callback(self, event=None):
        """Copy selected lines to the clipboard, with prompts.

        This makes the copied text useful for doc-tests and interactive
        shell code examples.

        This always copies entire lines, even if only part of the first
        and/or last lines is selected.
        """
        text = self.text
        selfirst = text.index('sel.first linestart')
        if selfirst is None:  # Should not be possible.
            return  # No selection, do nothing.
        sellast = text.index('sel.last')
        if sellast[-1] != '0':
            sellast = text.index("sel.last+1line linestart")
        text.clipboard_clear()
        prompt_text = self.get_prompt_text(selfirst, sellast)
        text.clipboard_append(prompt_text)

    reading = False
    executing = False
    canceled = False
    endoffile = False
    closing = False
    _stop_readline_flag = False

    def set_warning_stream(self, stream):
        global warning_stream
        warning_stream = stream

    def get_warning_stream(self):
        return warning_stream

    def toggle_debugger(self, event=None):
        if self.executing:
            messagebox.showerror("Don't debug now",
                "You can only toggle the debugger when idle",
                parent=self.text)
            self.set_debugger_indicator()
            return "break"
        else:
            db = self.interp.getdebugger()
            if db:
                self.close_debugger()
            else:
                self.open_debugger()

    def set_debugger_indicator(self):
        db = self.interp.getdebugger()
        self.setvar("<<toggle-debugger>>", not not db)

    def toggle_jit_stack_viewer(self, event=None):
        pass # All we need is the variable

    def close_debugger(self):
        db = self.interp.getdebugger()
        if db:
            self.interp.setdebugger(None)
            db.close()
            if self.interp.rpcclt:
                debugger_r.close_remote_debugger(self.interp.rpcclt)
            self.resetoutput()
            self.console.write("[DEBUG OFF]\n")
            self.prompt = self.sys_ps1
            self.showprompt()
        self.set_debugger_indicator()

    def open_debugger(self):
        if self.interp.rpcclt:
            dbg_gui = debugger_r.start_remote_debugger(self.interp.rpcclt,
                                                           self)
        else:
            dbg_gui = debugger.Debugger(self)
        self.interp.setdebugger(dbg_gui)
        dbg_gui.load_breakpoints()
        self.prompt = "[DEBUG ON]\n" + self.sys_ps1
        self.showprompt()
        self.set_debugger_indicator()

    def debug_menu_postcommand(self):
        state = 'disabled' if self.executing else 'normal'
        self.update_menu_state('debug', '*tack*iewer', state)

    def beginexecuting(self):
        "Helper for ModifiedInterpreter"
        self.resetoutput()
        self.executing = True

    def endexecuting(self):
        "Helper for ModifiedInterpreter"
        self.executing = False
        self.canceled = False
        self.showprompt()

    def close(self):
        "Extend EditorWindow.close()"
        if self.executing:
            response = messagebox.askokcancel(
                "Kill?",
                "Your program is still running!\n Do you want to kill it?",
                default="ok",
                parent=self.text)
            if response is False:
                return "cancel"
        self.stop_readline()
        self.canceled = True
        self.closing = True
        return EditorWindow.close(self)

    def _close(self):
        "Extend EditorWindow._close(), shut down debugger and execution server"
        self.close_debugger()
        if use_subprocess:
            self.interp.kill_subprocess()
        # Restore std streams
        sys.stdout = self.save_stdout
        sys.stderr = self.save_stderr
        sys.stdin = self.save_stdin
        # Break cycles
        self.interp = None
        self.console = None
        self.flist.pyshell = None
        self.history = None
        EditorWindow._close(self)

    def ispythonsource(self, filename):
        "Override EditorWindow method: never remove the colorizer"
        return True

    def short_title(self):
        return self.shell_title

    COPYRIGHT = \
          'Type "help", "copyright", "credits" or "license()" for more information.'

    def begin(self):
        self.text.mark_set("iomark", "insert")
        self.resetoutput()
        if use_subprocess:
            nosub = ''
            client = self.interp.start_subprocess()
            if not client:
                self.close()
                return False
        else:
            nosub = ("==== No Subprocess ====\n\n" +
                    "WARNING: Running IDLE without a Subprocess is deprecated\n" +
                    "and will be removed in a later version. See Help/IDLE Help\n" +
                    "for details.\n\n")
            sys.displayhook = rpc.displayhook

        self.write("Python %s on %s\n%s\n%s" %
                   (sys.version, sys.platform, self.COPYRIGHT, nosub))
        self.text.focus_force()
        self.showprompt()
        # User code should use separate default Tk root window
        import tkinter
        tkinter._support_default_root = True
        tkinter._default_root = None
        return True

    def stop_readline(self):
        if not self.reading:  # no nested mainloop to exit.
            return
        self._stop_readline_flag = True
        self.top.quit()

    def readline(self):
        save = self.reading
        try:
            self.reading = True
            self.top.mainloop()  # nested mainloop()
        finally:
            self.reading = save
        if self._stop_readline_flag:
            self._stop_readline_flag = False
            return ""
        line = self.text.get("iomark", "end-1c")
        if len(line) == 0:  # may be EOF if we quit our mainloop with Ctrl-C
            line = "\n"
        self.resetoutput()
        if self.canceled:
            self.canceled = False
            if not use_subprocess:
                raise KeyboardInterrupt
        if self.endoffile:
            self.endoffile = False
            line = ""
        return line

    def isatty(self):
        return True

    def cancel_callback(self, event=None):
        try:
            if self.text.compare("sel.first", "!=", "sel.last"):
                return # Active selection -- always use default binding
        except:
            pass
        if not (self.executing or self.reading):
            self.resetoutput()
            self.interp.write("KeyboardInterrupt\n")
            self.showprompt()
            return "break"
        self.endoffile = False
        self.canceled = True
        if (self.executing and self.interp.rpcclt):
            if self.interp.getdebugger():
                self.interp.restart_subprocess()
            else:
                self.interp.interrupt_subprocess()
        if self.reading:
            self.top.quit()  # exit the nested mainloop() in readline()
        return "break"

    def eof_callback(self, event):
        if self.executing and not self.reading:
            return # Let the default binding (delete next char) take over
        if not (self.text.compare("iomark", "==", "insert") and
                self.text.compare("insert", "==", "end-1c")):
            return # Let the default binding (delete next char) take over
        if not self.executing:
            self.resetoutput()
            self.close()
        else:
            self.canceled = False
            self.endoffile = True
            self.top.quit()
        return "break"

    def linefeed_callback(self, event):
        # Insert a linefeed without entering anything (still autoindented)
        if self.reading:
            self.text.insert("insert", "\n")
            self.text.see("insert")
        else:
            self.newline_and_indent_event(event)
        return "break"

    def enter_callback(self, event):
        if self.executing and not self.reading:
            return # Let the default binding (insert '\n') take over
        # If some text is selected, recall the selection
        # (but only if this before the I/O mark)
        try:
            sel = self.text.get("sel.first", "sel.last")
            if sel:
                if self.text.compare("sel.last", "<=", "iomark"):
                    self.recall(sel, event)
                    return "break"
        except:
            pass
        # If we're strictly before the line containing iomark, recall
        # the current line, less a leading prompt, less leading or
        # trailing whitespace
        if self.text.compare("insert", "<", "iomark linestart"):
            # Check if there's a relevant stdin range -- if so, use it.
            # Note: "stdin" blocks may include several successive statements,
            # so look for "console" tags on the newline before each statement
            # (and possibly on prompts).
            prev = self.text.tag_prevrange("stdin", "insert")
            if (
                    prev and
                    self.text.compare("insert", "<", prev[1]) and
                    # The following is needed to handle empty statements.
                    "console" not in self.text.tag_names("insert")
            ):
                prev_cons = self.text.tag_prevrange("console", "insert")
                if prev_cons and self.text.compare(prev_cons[1], ">=", prev[0]):
                    prev = (prev_cons[1], prev[1])
                next_cons = self.text.tag_nextrange("console", "insert")
                if next_cons and self.text.compare(next_cons[0], "<", prev[1]):
                    prev = (prev[0], self.text.index(next_cons[0] + "+1c"))
                self.recall(self.text.get(prev[0], prev[1]), event)
                return "break"
            next = self.text.tag_nextrange("stdin", "insert")
            if next and self.text.compare("insert lineend", ">=", next[0]):
                next_cons = self.text.tag_nextrange("console", "insert lineend")
                if next_cons and self.text.compare(next_cons[0], "<", next[1]):
                    next = (next[0], self.text.index(next_cons[0] + "+1c"))
                self.recall(self.text.get(next[0], next[1]), event)
                return "break"
            # No stdin mark -- just get the current line, less any prompt
            indices = self.text.tag_nextrange("console", "insert linestart")
            if indices and \
               self.text.compare(indices[0], "<=", "insert linestart"):
                self.recall(self.text.get(indices[1], "insert lineend"), event)
            else:
                self.recall(self.text.get("insert linestart", "insert lineend"), event)
            return "break"
        # If we're between the beginning of the line and the iomark, i.e.
        # in the prompt area, move to the end of the prompt
        if self.text.compare("insert", "<", "iomark"):
            self.text.mark_set("insert", "iomark")
        # If we're in the current input and there's only whitespace
        # beyond the cursor, erase that whitespace first
        s = self.text.get("insert", "end-1c")
        if s and not s.strip():
            self.text.delete("insert", "end-1c")
        # If we're in the current input before its last line,
        # insert a newline right at the insert point
        if self.text.compare("insert", "<", "end-1c linestart"):
            self.newline_and_indent_event(event)
            return "break"
        # We're in the last line; append a newline and submit it
        self.text.mark_set("insert", "end-1c")
        if self.reading:
            self.text.insert("insert", "\n")
            self.text.see("insert")
        else:
            self.newline_and_indent_event(event)
        self.text.update_idletasks()
        if self.reading:
            self.top.quit() # Break out of recursive mainloop()
        else:
            self.runit()
        return "break"

    def recall(self, s, event):
        # remove leading and trailing empty or whitespace lines
        s = re.sub(r'^\s*\n', '', s)
        s = re.sub(r'\n\s*$', '', s)
        lines = s.split('\n')
        self.text.undo_block_start()
        try:
            self.text.tag_remove("sel", "1.0", "end")
            self.text.mark_set("insert", "end-1c")
            prefix = self.text.get("insert linestart", "insert")
            if prefix.rstrip().endswith(':'):
                self.newline_and_indent_event(event)
                prefix = self.text.get("insert linestart", "insert")
            self.text.insert("insert", lines[0].strip(),
                             self.user_input_insert_tags)
            if len(lines) > 1:
                orig_base_indent = re.search(r'^([ \t]*)', lines[0]).group(0)
                new_base_indent  = re.search(r'^([ \t]*)', prefix).group(0)
                for line in lines[1:]:
                    if line.startswith(orig_base_indent):
                        # replace orig base indentation with new indentation
                        line = new_base_indent + line[len(orig_base_indent):]
                    self.text.insert('insert', '\n' + line.rstrip(),
                                     self.user_input_insert_tags)
        finally:
            self.text.see("insert")
            self.text.undo_block_stop()

    _last_newline_re = re.compile(r"[ \t]*(\n[ \t]*)?\Z")
    def runit(self):
        index_before = self.text.index("end-2c")
        line = self.text.get("iomark", "end-1c")
        # Strip off last newline and surrounding whitespace.
        # (To allow you to hit return twice to end a statement.)
        line = self._last_newline_re.sub("", line)
        input_is_complete = self.interp.runsource(line)
        if not input_is_complete:
            if self.text.get(index_before) == '\n':
                self.text.tag_remove(self.user_input_insert_tags, index_before)
            self.shell_sidebar.update_sidebar()

    def open_stack_viewer(self, event=None):  # -n mode only
        if self.interp.rpcclt:
            return self.interp.remote_stack_viewer()

        from idlelib.stackviewer import StackBrowser
        try:
            StackBrowser(self.root, sys.last_value, self.flist)
        except:
            messagebox.showerror("No stack trace",
                "There is no stack trace yet.\n"
                "(sys.last_value is not defined)",
                parent=self.text)
        return None

    def view_restart_mark(self, event=None):
        self.text.see("iomark")
        self.text.see("restart")

    def restart_shell(self, event=None):
        "Callback for Run/Restart Shell Cntl-F6"
        self.interp.restart_subprocess(with_cwd=True)

    def showprompt(self):
        self.resetoutput()

        prompt = self.prompt
        if self.sys_ps1 and prompt.endswith(self.sys_ps1):
            prompt = prompt[:-len(self.sys_ps1)]
        self.text.tag_add("console", "iomark-1c")
        self.console.write(prompt)

        self.shell_sidebar.update_sidebar()
        self.text.mark_set("insert", "end-1c")
        self.set_line_and_column()
        self.io.reset_undo()

    def show_warning(self, msg):
        width = self.interp.tkconsole.width
        wrapper = TextWrapper(width=width, tabsize=8, expand_tabs=True)
        wrapped_msg = '\n'.join(wrapper.wrap(msg))
        if not wrapped_msg.endswith('\n'):
            wrapped_msg += '\n'
        self.per.bottom.insert("iomark linestart", wrapped_msg, "stderr")

    def resetoutput(self):
        source = self.text.get("iomark", "end-1c")
        if self.history:
            self.history.store(source)
        if self.text.get("end-2c") != "\n":
            self.text.insert("end-1c", "\n")
        self.text.mark_set("iomark", "end-1c")
        self.set_line_and_column()
        self.ctip.remove_calltip_window()

    def write(self, s, tags=()):
        try:
            self.text.mark_gravity("iomark", "right")
            count = OutputWindow.write(self, s, tags, "iomark")
            self.text.mark_gravity("iomark", "left")
        except:
            raise ###pass  # ### 11Aug07 KBK if we are expecting exceptions
                           # let's find out what they are and be specific.
        if self.canceled:
            self.canceled = False
            if not use_subprocess:
                raise KeyboardInterrupt
        return count

    def rmenu_check_cut(self):
        try:
            if self.text.compare('sel.first', '<', 'iomark'):
                return 'disabled'
        except TclError: # no selection, so the index 'sel.first' doesn't exist
            return 'disabled'
        return super().rmenu_check_cut()

    def rmenu_check_paste(self):
        if self.text.compare('insert','<','iomark'):
            return 'disabled'
        return super().rmenu_check_paste()

    def squeeze_current_text_event(self, event=None):
        self.squeezer.squeeze_current_text()
        self.shell_sidebar.update_sidebar()

    def on_squeezed_expand(self, index, text, tags):
        self.shell_sidebar.update_sidebar()


def fix_x11_paste(root):
    "Make paste replace selection on x11.  See issue #5124."
    if root._windowingsystem == 'x11':
        for cls in 'Text', 'Entry', 'Spinbox':
            root.bind_class(
                cls,
                '<<Paste>>',
                'catch {%W delete sel.first sel.last}\n' +
                        root.bind_class(cls, '<<Paste>>'))


usage_msg = """\

USAGE: idle  [-deins] [-t title] [file]*
       idle  [-dns] [-t title] (-c cmd | -r file) [arg]*
       idle  [-dns] [-t title] - [arg]*

  -h         print this help message and exit
  -n         run IDLE without a subprocess (DEPRECATED,
             see Help/IDLE Help for details)

The following options will override the IDLE 'settings' configuration:

  -e         open an edit window
  -i         open a shell window

The following options imply -i and will open a shell:

  -c cmd     run the command in a shell, or
  -r file    run script from file

  -d         enable the debugger
  -s         run $IDLESTARTUP or $PYTHONSTARTUP before anything else
  -t title   set title of shell window

A default edit window will be bypassed when -c, -r, or - are used.

[arg]* are passed to the command (-c) or script (-r) in sys.argv[1:].

Examples:

idle
        Open an edit window or shell depending on IDLE's configuration.

idle foo.py foobar.py
        Edit the files, also open a shell if configured to start with shell.

idle -est "Baz" foo.py
        Run $IDLESTARTUP or $PYTHONSTARTUP, edit foo.py, and open a shell
        window with the title "Baz".

idle -c "import sys; print(sys.argv)" "foo"
        Open a shell window and run the command, passing "-c" in sys.argv[0]
        and "foo" in sys.argv[1].

idle -d -s -r foo.py "Hello World"
        Open a shell window, run a startup script, enable the debugger, and
        run foo.py, passing "foo.py" in sys.argv[0] and "Hello World" in
        sys.argv[1].

echo "import sys; print(sys.argv)" | idle - "foobar"
        Open a shell window, run the script piped in, passing '' in sys.argv[0]
        and "foobar" in sys.argv[1].
"""

def main():
    import getopt
    from platform import system
    from idlelib import testing  # bool value
    from idlelib import macosx

    global flist, root, use_subprocess

    capture_warnings(True)
    use_subprocess = True
    enable_shell = False
    enable_edit = False
    debug = False
    cmd = None
    script = None
    startup = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:deihnr:st:")
    except getopt.error as msg:
        print(f"Error: {msg}\n{usage_msg}", file=sys.stderr)
        sys.exit(2)
    for o, a in opts:
        if o == '-c':
            cmd = a
            enable_shell = True
        if o == '-d':
            debug = True
            enable_shell = True
        if o == '-e':
            enable_edit = True
        if o == '-h':
            sys.stdout.write(usage_msg)
            sys.exit()
        if o == '-i':
            enable_shell = True
        if o == '-n':
            print(" Warning: running IDLE without a subprocess is deprecated.",
                  file=sys.stderr)
            use_subprocess = False
        if o == '-r':
            script = a
            if os.path.isfile(script):
                pass
            else:
                print("No script file: ", script)
                sys.exit()
            enable_shell = True
        if o == '-s':
            startup = True
            enable_shell = True
        if o == '-t':
            PyShell.shell_title = a
            enable_shell = True
    if args and args[0] == '-':
        cmd = sys.stdin.read()
        enable_shell = True
    # process sys.argv and sys.path:
    for i in range(len(sys.path)):
        sys.path[i] = os.path.abspath(sys.path[i])
    if args and args[0] == '-':
        sys.argv = [''] + args[1:]
    elif cmd:
        sys.argv = ['-c'] + args
    elif script:
        sys.argv = [script] + args
    elif args:
        enable_edit = True
        pathx = []
        for filename in args:
            pathx.append(os.path.dirname(filename))
        for dir in pathx:
            dir = os.path.abspath(dir)
            if not dir in sys.path:
                sys.path.insert(0, dir)
    else:
        dir = os.getcwd()
        if dir not in sys.path:
            sys.path.insert(0, dir)
    # check the IDLE settings configuration (but command line overrides)
    edit_start = idleConf.GetOption('main', 'General',
                                    'editor-on-startup', type='bool')
    enable_edit = enable_edit or edit_start
    enable_shell = enable_shell or not enable_edit

    # Setup root.  Don't break user code run in IDLE process.
    # Don't change environment when testing.
    if use_subprocess and not testing:
        NoDefaultRoot()
    root = Tk(className="Idle")
    root.withdraw()
    from idlelib.run import fix_scaling
    fix_scaling(root)

    # set application icon
    icondir = os.path.join(os.path.dirname(__file__), 'Icons')
    if system() == 'Windows':
        iconfile = os.path.join(icondir, 'idle.ico')
        root.wm_iconbitmap(default=iconfile)
    elif not macosx.isAquaTk():
        if TkVersion >= 8.6:
            ext = '.png'
            sizes = (16, 32, 48, 256)
        else:
            ext = '.gif'
            sizes = (16, 32, 48)
        iconfiles = [os.path.join(icondir, 'idle_%d%s' % (size, ext))
                     for size in sizes]
        icons = [PhotoImage(master=root, file=iconfile)
                 for iconfile in iconfiles]
        root.wm_iconphoto(True, *icons)

    # start editor and/or shell windows:
    fixwordbreaks(root)
    fix_x11_paste(root)
    flist = PyShellFileList(root)
    macosx.setupApp(root, flist)

    if enable_edit:
        if not (cmd or script):
            for filename in args[:]:
                if flist.open(filename) is None:
                    # filename is a directory actually, disconsider it
                    args.remove(filename)
            if not args:
                flist.new()

    if enable_shell:
        shell = flist.open_shell()
        if not shell:
            return # couldn't open shell
        if macosx.isAquaTk() and flist.dict:
            # On OSX: when the user has double-clicked on a file that causes
            # IDLE to be launched the shell window will open just in front of
            # the file she wants to see. Lower the interpreter window when
            # there are open files.
            shell.top.lower()
    else:
        shell = flist.pyshell

    # Handle remaining options. If any of these are set, enable_shell
    # was set also, so shell must be true to reach here.
    if debug:
        shell.open_debugger()
    if startup:
        filename = os.environ.get("IDLESTARTUP") or \
                   os.environ.get("PYTHONSTARTUP")
        if filename and os.path.isfile(filename):
            shell.interp.execfile(filename)
    if cmd or script:
        shell.interp.runcommand("""if 1:
            import sys as _sys
            _sys.argv = {!r}
            del _sys
            \n""".format(sys.argv))
        if cmd:
            shell.interp.execsource(cmd)
        elif script:
            shell.interp.prepend_syspath(script)
            shell.interp.execfile(script)
    elif shell:
        # If there is a shell window and no cmd or script in progress,
        # check for problematic issues and print warning message(s) in
        # the IDLE shell window; this is less intrusive than always
        # opening a separate window.

        # Warn if the "Prefer tabs when opening documents" system
        # preference is set to "Always".
        prefer_tabs_preference_warning = macosx.preferTabsPreferenceWarning()
        if prefer_tabs_preference_warning:
            shell.show_warning(prefer_tabs_preference_warning)

    while flist.inversedict:  # keep IDLE running while files are open.
        root.mainloop()
    root.destroy()
    capture_warnings(False)


if __name__ == "__main__":
    main()

capture_warnings(False)  # Make sure turned off; see issue 18081
