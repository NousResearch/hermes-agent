""" idlelib.run

Simplified, pyshell.ModifiedInterpreter spawns a subprocess with
f'''{sys.executable} -c "__import__('idlelib.run').run.main()"'''
'.run' is needed because __import__ returns idlelib, not idlelib.run.
"""
import contextlib
import functools
import io
import linecache
import queue
import sys
import textwrap
import time
import traceback
import _thread as thread
import threading
import warnings

import idlelib  # testing
from idlelib import autocomplete  # AutoComplete, fetch_encodings
from idlelib import calltip  # Calltip
from idlelib import debugger_r  # start_debugger
from idlelib import debugobj_r  # remote_object_tree_item
from idlelib import iomenu  # encoding
from idlelib import rpc  # multiple objects
from idlelib import stackviewer  # StackTreeItem
import __main__

import tkinter  # Use tcl and, if startup fails, messagebox.
if not hasattr(sys.modules['idlelib.run'], 'firstrun'):
    # Undo modifications of tkinter by idlelib imports; see bpo-25507.
    for mod in ('simpledialog', 'messagebox', 'font',
                'dialog', 'filedialog', 'commondialog',
                'ttk'):
        delattr(tkinter, mod)
        del sys.modules['tkinter.' + mod]
    # Avoid AttributeError if run again; see bpo-37038.
    sys.modules['idlelib.run'].firstrun = False

LOCALHOST = '127.0.0.1'

try:
    eof = 'Ctrl-D (end-of-file)'
    exit.eof = eof
    quit.eof = eof
except NameError: # In case subprocess started with -S (maybe in future).
    pass


def idle_formatwarning(message, category, filename, lineno, line=None):
    """Format warnings the IDLE way."""

    s = "\nWarning (from warnings module):\n"
    s += f'  File \"{filename}\", line {lineno}\n'
    if line is None:
        line = linecache.getline(filename, lineno)
    line = line.strip()
    if line:
        s += "    %s\n" % line
    s += f"{category.__name__}: {message}\n"
    return s

def idle_showwarning_subproc(
        message, category, filename, lineno, file=None, line=None):
    """Show Idle-format warning after replacing warnings.showwarning.

    The only difference is the formatter called.
    """
    if file is None:
        file = sys.stderr
    try:
        file.write(idle_formatwarning(
                message, category, filename, lineno, line))
    except OSError:
        pass # the file (probably stderr) is invalid - this warning gets lost.

_warnings_showwarning = None

def capture_warnings(capture):
    "Replace warning.showwarning with idle_showwarning_subproc, or reverse."

    global _warnings_showwarning
    if capture:
        if _warnings_showwarning is None:
            _warnings_showwarning = warnings.showwarning
            warnings.showwarning = idle_showwarning_subproc
    else:
        if _warnings_showwarning is not None:
            warnings.showwarning = _warnings_showwarning
            _warnings_showwarning = None

capture_warnings(True)
tcl = tkinter.Tcl()

def handle_tk_events(tcl=tcl):
    """Process any tk events that are ready to be dispatched if tkinter
    has been imported, a tcl interpreter has been created and tk has been
    loaded."""
    tcl.eval("update")

# Thread shared globals: Establish a queue between a subthread (which handles
# the socket) and the main thread (which runs user code), plus global
# completion, exit and interruptable (the main thread) flags:

exit_now = False
quitting = False
interruptable = False

def main(del_exitfunc=False):
    """Start the Python execution server in a subprocess

    In the Python subprocess, RPCServer is instantiated with handlerclass
    MyHandler, which inherits register/unregister methods from RPCHandler via
    the mix-in class SocketIO.

    When the RPCServer 'server' is instantiated, the TCPServer initialization
    creates an instance of run.MyHandler and calls its handle() method.
    handle() instantiates a run.Executive object, passing it a reference to the
    MyHandler object.  That reference is saved as attribute rpchandler of the
    Executive instance.  The Executive methods have access to the reference and
    can pass it on to entities that they command
    (e.g. debugger_r.Debugger.start_debugger()).  The latter, in turn, can
    call MyHandler(SocketIO) register/unregister methods via the reference to
    register and unregister themselves.

    """
    global exit_now
    global quitting
    global no_exitfunc
    no_exitfunc = del_exitfunc
    #time.sleep(15) # test subprocess not responding
    try:
        assert(len(sys.argv) > 1)
        port = int(sys.argv[-1])
    except:
        print("IDLE Subprocess: no IP port passed in sys.argv.",
              file=sys.__stderr__)
        return

    capture_warnings(True)
    sys.argv[:] = [""]
    threading.Thread(target=manage_socket,
                     name='SockThread',
                     args=((LOCALHOST, port),),
                     daemon=True,
                    ).start()

    while True:
        try:
            if exit_now:
                try:
                    exit()
                except KeyboardInterrupt:
                    # exiting but got an extra KBI? Try again!
                    continue
            try:
                request = rpc.request_queue.get(block=True, timeout=0.05)
            except queue.Empty:
                request = None
                # Issue 32207: calling handle_tk_events here adds spurious
                # queue.Empty traceback to event handling exceptions.
            if request:
                seq, (method, args, kwargs) = request
                ret = method(*args, **kwargs)
                rpc.response_queue.put((seq, ret))
            else:
                handle_tk_events()
        except KeyboardInterrupt:
            if quitting:
                exit_now = True
            continue
        except SystemExit:
            capture_warnings(False)
            raise
        except:
            type, value, tb = sys.exc_info()
            try:
                print_exception()
                rpc.response_queue.put((seq, None))
            except:
                # Link didn't work, print same exception to __stderr__
                traceback.print_exception(type, value, tb, file=sys.__stderr__)
                exit()
            else:
                continue

def manage_socket(address):
    for i in range(3):
        time.sleep(i)
        try:
            server = MyRPCServer(address, MyHandler)
            break
        except OSError as err:
            print("IDLE Subprocess: OSError: " + err.args[1] +
                  ", retrying....", file=sys.__stderr__)
            socket_error = err
    else:
        print("IDLE Subprocess: Connection to "
              "IDLE GUI failed, exiting.", file=sys.__stderr__)
        show_socket_error(socket_error, address)
        global exit_now
        exit_now = True
        return
    server.handle_request() # A single request only

def show_socket_error(err, address):
    "Display socket error from manage_socket."
    import tkinter
    from tkinter.messagebox import showerror
    root = tkinter.Tk()
    fix_scaling(root)
    root.withdraw()
    showerror(
            "Subprocess Connection Error",
            f"IDLE's subprocess can't connect to {address[0]}:{address[1]}.\n"
            f"Fatal OSError #{err.errno}: {err.strerror}.\n"
            "See the 'Startup failure' section of the IDLE doc, online at\n"
            "https://docs.python.org/3/library/idle.html#startup-failure",
            parent=root)
    root.destroy()


def get_message_lines(typ, exc, tb):
    "Return line composing the exception message."
    if typ in (AttributeError, NameError):
        # 3.10+ hints are not directly accessible from python (#44026).
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            sys.__excepthook__(typ, exc, tb)
        return [err.getvalue().split("\n")[-2] + "\n"]
    else:
        return traceback.format_exception_only(typ, exc)


def print_exception():
    import linecache
    linecache.checkcache()
    flush_stdout()
    efile = sys.stderr
    typ, val, tb = excinfo = sys.exc_info()
    sys.last_type, sys.last_value, sys.last_traceback = excinfo
    seen = set()

    def print_exc(typ, exc, tb):
        seen.add(id(exc))
        context = exc.__context__
        cause = exc.__cause__
        if cause is not None and id(cause) not in seen:
            print_exc(type(cause), cause, cause.__traceback__)
            print("\nThe above exception was the direct cause "
                  "of the following exception:\n", file=efile)
        elif (context is not None and
              not exc.__suppress_context__ and
              id(context) not in seen):
            print_exc(type(context), context, context.__traceback__)
            print("\nDuring handling of the above exception, "
                  "another exception occurred:\n", file=efile)
        if tb:
            tbe = traceback.extract_tb(tb)
            print('Traceback (most recent call last):', file=efile)
            exclude = ("run.py", "rpc.py", "threading.py", "queue.py",
                       "debugger_r.py", "bdb.py")
            cleanup_traceback(tbe, exclude)
            traceback.print_list(tbe, file=efile)
        lines = get_message_lines(typ, exc, tb)
        for line in lines:
            print(line, end='', file=efile)

    print_exc(typ, val, tb)

def cleanup_traceback(tb, exclude):
    "Remove excluded traces from beginning/end of tb; get cached lines"
    orig_tb = tb[:]
    while tb:
        for rpcfile in exclude:
            if tb[0][0].count(rpcfile):
                break    # found an exclude, break for: and delete tb[0]
        else:
            break        # no excludes, have left RPC code, break while:
        del tb[0]
    while tb:
        for rpcfile in exclude:
            if tb[-1][0].count(rpcfile):
                break
        else:
            break
        del tb[-1]
    if len(tb) == 0:
        # exception was in IDLE internals, don't prune!
        tb[:] = orig_tb[:]
        print("** IDLE Internal Exception: ", file=sys.stderr)
    rpchandler = rpc.objecttable['exec'].rpchandler
    for i in range(len(tb)):
        fn, ln, nm, line = tb[i]
        if nm == '?':
            nm = "-toplevel-"
        if not line and fn.startswith("<pyshell#"):
            line = rpchandler.remotecall('linecache', 'getline',
                                              (fn, ln), {})
        tb[i] = fn, ln, nm, line

def flush_stdout():
    """XXX How to do this now?"""

def exit():
    """Exit subprocess, possibly after first clearing exit functions.

    If config-main.cfg/.def 'General' 'delete-exitfunc' is True, then any
    functions registered with atexit will be removed before exiting.
    (VPython support)

    """
    if no_exitfunc:
        import atexit
        atexit._clear()
    capture_warnings(False)
    sys.exit(0)


def fix_scaling(root):
    """Scale fonts on HiDPI displays."""
    import tkinter.font
    scaling = float(root.tk.call('tk', 'scaling'))
    if scaling > 1.4:
        for name in tkinter.font.names(root):
            font = tkinter.font.Font(root=root, name=name, exists=True)
            size = int(font['size'])
            if size < 0:
                font['size'] = round(-0.75*size)


def fixdoc(fun, text):
    tem = (fun.__doc__ + '\n\n') if fun.__doc__ is not None else ''
    fun.__doc__ = tem + textwrap.fill(textwrap.dedent(text))

RECURSIONLIMIT_DELTA = 30

def install_recursionlimit_wrappers():
    """Install wrappers to always add 30 to the recursion limit."""
    # see: bpo-26806

    @functools.wraps(sys.setrecursionlimit)
    def setrecursionlimit(*args, **kwargs):
        # mimic the original sys.setrecursionlimit()'s input handling
        if kwargs:
            raise TypeError(
                "setrecursionlimit() takes no keyword arguments")
        try:
            limit, = args
        except ValueError:
            raise TypeError(f"setrecursionlimit() takes exactly one "
                            f"argument ({len(args)} given)")
        if not limit > 0:
            raise ValueError(
                "recursion limit must be greater or equal than 1")

        return setrecursionlimit.__wrapped__(limit + RECURSIONLIMIT_DELTA)

    fixdoc(setrecursionlimit, f"""\
            This IDLE wrapper adds {RECURSIONLIMIT_DELTA} to prevent possible
            uninterruptible loops.""")

    @functools.wraps(sys.getrecursionlimit)
    def getrecursionlimit():
        return getrecursionlimit.__wrapped__() - RECURSIONLIMIT_DELTA

    fixdoc(getrecursionlimit, f"""\
            This IDLE wrapper subtracts {RECURSIONLIMIT_DELTA} to compensate
            for the {RECURSIONLIMIT_DELTA} IDLE adds when setting the limit.""")

    # add the delta to the default recursion limit, to compensate
    sys.setrecursionlimit(sys.getrecursionlimit() + RECURSIONLIMIT_DELTA)

    sys.setrecursionlimit = setrecursionlimit
    sys.getrecursionlimit = getrecursionlimit


def uninstall_recursionlimit_wrappers():
    """Uninstall the recursion limit wrappers from the sys module.

    IDLE only uses this for tests. Users can import run and call
    this to remove the wrapping.
    """
    if (
            getattr(sys.setrecursionlimit, '__wrapped__', None) and
            getattr(sys.getrecursionlimit, '__wrapped__', None)
    ):
        sys.setrecursionlimit = sys.setrecursionlimit.__wrapped__
        sys.getrecursionlimit = sys.getrecursionlimit.__wrapped__
        sys.setrecursionlimit(sys.getrecursionlimit() - RECURSIONLIMIT_DELTA)


class MyRPCServer(rpc.RPCServer):

    def handle_error(self, request, client_address):
        """Override RPCServer method for IDLE

        Interrupt the MainThread and exit server if link is dropped.

        """
        global quitting
        try:
            raise
        except SystemExit:
            raise
        except EOFError:
            global exit_now
            exit_now = True
            thread.interrupt_main()
        except:
            erf = sys.__stderr__
            print(textwrap.dedent(f"""
            {'-'*40}
            Unhandled exception in user code execution server!'
            Thread: {threading.current_thread().name}
            IDLE Client Address: {client_address}
            Request: {request!r}
            """), file=erf)
            traceback.print_exc(limit=-20, file=erf)
            print(textwrap.dedent(f"""
            *** Unrecoverable, server exiting!

            Users should never see this message; it is likely transient.
            If this recurs, report this with a copy of the message
            and an explanation of how to make it repeat.
            {'-'*40}"""), file=erf)
            quitting = True
            thread.interrupt_main()


# Pseudofiles for shell-remote communication (also used in pyshell)

class StdioFile(io.TextIOBase):

    def __init__(self, shell, tags, encoding='utf-8', errors='strict'):
        self.shell = shell
        self.tags = tags
        self._encoding = encoding
        self._errors = errors

    @property
    def encoding(self):
        return self._encoding

    @property
    def errors(self):
        return self._errors

    @property
    def name(self):
        return '<%s>' % self.tags

    def isatty(self):
        return True


class StdOutputFile(StdioFile):

    def writable(self):
        return True

    def write(self, s):
        if self.closed:
            raise ValueError("write to closed file")
        s = str.encode(s, self.encoding, self.errors).decode(self.encoding, self.errors)
        return self.shell.write(s, self.tags)


class StdInputFile(StdioFile):
    _line_buffer = ''

    def readable(self):
        return True

    def read(self, size=-1):
        if self.closed:
            raise ValueError("read from closed file")
        if size is None:
            size = -1
        elif not isinstance(size, int):
            raise TypeError('must be int, not ' + type(size).__name__)
        result = self._line_buffer
        self._line_buffer = ''
        if size < 0:
            while line := self.shell.readline():
                result += line
        else:
            while len(result) < size:
                line = self.shell.readline()
                if not line: break
                result += line
            self._line_buffer = result[size:]
            result = result[:size]
        return result

    def readline(self, size=-1):
        if self.closed:
            raise ValueError("read from closed file")
        if size is None:
            size = -1
        elif not isinstance(size, int):
            raise TypeError('must be int, not ' + type(size).__name__)
        line = self._line_buffer or self.shell.readline()
        if size < 0:
            size = len(line)
        eol = line.find('\n', 0, size)
        if eol >= 0:
            size = eol + 1
        self._line_buffer = line[size:]
        return line[:size]

    def close(self):
        self.shell.close()


class MyHandler(rpc.RPCHandler):

    def handle(self):
        """Override base method"""
        executive = Executive(self)
        self.register("exec", executive)
        self.console = self.get_remote_proxy("console")
        sys.stdin = StdInputFile(self.console, "stdin",
                                 iomenu.encoding, iomenu.errors)
        sys.stdout = StdOutputFile(self.console, "stdout",
                                   iomenu.encoding, iomenu.errors)
        sys.stderr = StdOutputFile(self.console, "stderr",
                                   iomenu.encoding, "backslashreplace")

        sys.displayhook = rpc.displayhook
        # page help() text to shell.
        import pydoc # import must be done here to capture i/o binding
        pydoc.pager = pydoc.plainpager

        # Keep a reference to stdin so that it won't try to exit IDLE if
        # sys.stdin gets changed from within IDLE's shell. See issue17838.
        self._keep_stdin = sys.stdin

        install_recursionlimit_wrappers()

        self.interp = self.get_remote_proxy("interp")
        rpc.RPCHandler.getresponse(self, myseq=None, wait=0.05)

    def exithook(self):
        "override SocketIO method - wait for MainThread to shut us down"
        time.sleep(10)

    def EOFhook(self):
        "Override SocketIO method - terminate wait on callback and exit thread"
        global quitting
        quitting = True
        thread.interrupt_main()

    def decode_interrupthook(self):
        "interrupt awakened thread"
        global quitting
        quitting = True
        thread.interrupt_main()


class Executive:

    def __init__(self, rpchandler):
        self.rpchandler = rpchandler
        if idlelib.testing is False:
            self.locals = __main__.__dict__
            self.calltip = calltip.Calltip()
            self.autocomplete = autocomplete.AutoComplete()
        else:
            self.locals = {}

    def runcode(self, code):
        global interruptable
        try:
            self.user_exc_info = None
            interruptable = True
            try:
                exec(code, self.locals)
            finally:
                interruptable = False
        except SystemExit as e:
            if e.args:  # SystemExit called with an argument.
                ob = e.args[0]
                if not isinstance(ob, (type(None), int)):
                    print('SystemExit: ' + str(ob), file=sys.stderr)
            # Return to the interactive prompt.
        except:
            self.user_exc_info = sys.exc_info()  # For testing, hook, viewer.
            if quitting:
                exit()
            if sys.excepthook is sys.__excepthook__:
                print_exception()
            else:
                try:
                    sys.excepthook(*self.user_exc_info)
                except:
                    self.user_exc_info = sys.exc_info()  # For testing.
                    print_exception()
            jit = self.rpchandler.console.getvar("<<toggle-jit-stack-viewer>>")
            if jit:
                self.rpchandler.interp.open_remote_stack_viewer()
        else:
            flush_stdout()

    def interrupt_the_server(self):
        if interruptable:
            thread.interrupt_main()

    def start_the_debugger(self, gui_adap_oid):
        return debugger_r.start_debugger(self.rpchandler, gui_adap_oid)

    def stop_the_debugger(self, idb_adap_oid):
        "Unregister the Idb Adapter.  Link objects and Idb then subject to GC"
        self.rpchandler.unregister(idb_adap_oid)

    def get_the_calltip(self, name):
        return self.calltip.fetch_tip(name)

    def get_the_completion_list(self, what, mode):
        return self.autocomplete.fetch_completions(what, mode)

    def stackviewer(self, flist_oid=None):
        if self.user_exc_info:
            _, exc, tb = self.user_exc_info
        else:
            return None
        flist = None
        if flist_oid is not None:
            flist = self.rpchandler.get_remote_proxy(flist_oid)
        while tb and tb.tb_frame.f_globals["__name__"] in ["rpc", "run"]:
            tb = tb.tb_next
        exc.__traceback__ = tb
        item = stackviewer.StackTreeItem(exc, flist)
        return debugobj_r.remote_object_tree_item(item)


if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_run', verbosity=2)

capture_warnings(False)  # Make sure turned off; see bpo-18081.
