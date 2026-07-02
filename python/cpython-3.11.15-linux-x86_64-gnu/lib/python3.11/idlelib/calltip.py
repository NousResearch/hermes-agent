"""Pop up a reminder of how to call a function.

Call Tips are floating windows which display function, class, and method
parameter and docstring information when you type an opening parenthesis, and
which disappear when you type a closing parenthesis.
"""
import __main__
import inspect
import re
import sys
import textwrap
import types

from idlelib import calltip_w
from idlelib.hyperparser import HyperParser


class Calltip:

    def __init__(self, editwin=None):
        if editwin is None:  # subprocess and test
            self.editwin = None
        else:
            self.editwin = editwin
            self.text = editwin.text
            self.active_calltip = None
            self._calltip_window = self._make_tk_calltip_window

    def close(self):
        self._calltip_window = None

    def _make_tk_calltip_window(self):
        # See __init__ for usage
        return calltip_w.CalltipWindow(self.text)

    def remove_calltip_window(self, event=None):
        if self.active_calltip:
            self.active_calltip.hidetip()
            self.active_calltip = None

    def force_open_calltip_event(self, event):
        "The user selected the menu entry or hotkey, open the tip."
        self.open_calltip(True)
        return "break"

    def try_open_calltip_event(self, event):
        """Happens when it would be nice to open a calltip, but not really
        necessary, for example after an opening bracket, so function calls
        won't be made.
        """
        self.open_calltip(False)

    def refresh_calltip_event(self, event):
        if self.active_calltip and self.active_calltip.tipwindow:
            self.open_calltip(False)

    def open_calltip(self, evalfuncs):
        """Maybe close an existing calltip and maybe open a new calltip.

        Called from (force_open|try_open|refresh)_calltip_event functions.
        """
        hp = HyperParser(self.editwin, "insert")
        sur_paren = hp.get_surrounding_brackets('(')

        # If not inside parentheses, no calltip.
        if not sur_paren:
            self.remove_calltip_window()
            return

        # If a calltip is shown for the current parentheses, do
        # nothing.
        if self.active_calltip:
            opener_line, opener_col = map(int, sur_paren[0].split('.'))
            if (
                (opener_line, opener_col) ==
                (self.active_calltip.parenline, self.active_calltip.parencol)
            ):
                return

        hp.set_index(sur_paren[0])
        try:
            expression = hp.get_expression()
        except ValueError:
            expression = None
        if not expression:
            # No expression before the opening parenthesis, e.g.
            # because it's in a string or the opener for a tuple:
            # Do nothing.
            return

        # At this point, the current index is after an opening
        # parenthesis, in a section of code, preceded by a valid
        # expression. If there is a calltip shown, it's not for the
        # same index and should be closed.
        self.remove_calltip_window()

        # Simple, fast heuristic: If the preceding expression includes
        # an opening parenthesis, it likely includes a function call.
        if not evalfuncs and (expression.find('(') != -1):
            return

        argspec = self.fetch_tip(expression)
        if not argspec:
            return
        self.active_calltip = self._calltip_window()
        self.active_calltip.showtip(argspec, sur_paren[0], sur_paren[1])

    def fetch_tip(self, expression):
        """Return the argument list and docstring of a function or class.

        If there is a Python subprocess, get the calltip there.  Otherwise,
        either this fetch_tip() is running in the subprocess or it was
        called in an IDLE running without the subprocess.

        The subprocess environment is that of the most recently run script.  If
        two unrelated modules are being edited some calltips in the current
        module may be inoperative if the module was not the last to run.

        To find methods, fetch_tip must be fed a fully qualified name.

        """
        try:
            rpcclt = self.editwin.flist.pyshell.interp.rpcclt
        except AttributeError:
            rpcclt = None
        if rpcclt:
            return rpcclt.remotecall("exec", "get_the_calltip",
                                     (expression,), {})
        else:
            return get_argspec(get_entity(expression))


def get_entity(expression):
    """Return the object corresponding to expression evaluated
    in a namespace spanning sys.modules and __main.dict__.
    """
    if expression:
        namespace = {**sys.modules, **__main__.__dict__}
        try:
            return eval(expression, namespace)  # Only protect user code.
        except BaseException:
            # An uncaught exception closes idle, and eval can raise any
            # exception, especially if user classes are involved.
            return None

# The following are used in get_argspec and some in tests
_MAX_COLS = 85
_MAX_LINES = 5  # enough for bytes
_INDENT = ' '*4  # for wrapped signatures
_first_param = re.compile(r'(?<=\()\w*\,?\s*')
_default_callable_argspec = "See source or doc"
_invalid_method = "invalid method signature"

def get_argspec(ob):
    '''Return a string describing the signature of a callable object, or ''.

    For Python-coded functions and methods, the first line is introspected.
    Delete 'self' parameter for classes (.__init__) and bound methods.
    The next lines are the first lines of the doc string up to the first
    empty line or _MAX_LINES.    For builtins, this typically includes
    the arguments in addition to the return value.
    '''
    # Determine function object fob to inspect.
    try:
        ob_call = ob.__call__
    except BaseException:  # Buggy user object could raise anything.
        return ''  # No popup for non-callables.
    # For Get_argspecTest.test_buggy_getattr_class, CallA() & CallB().
    fob = ob_call if isinstance(ob_call, types.MethodType) else ob

    # Initialize argspec and wrap it to get lines.
    try:
        argspec = str(inspect.signature(fob))
    except Exception as err:
        msg = str(err)
        if msg.startswith(_invalid_method):
            return _invalid_method
        else:
            argspec = ''

    if isinstance(fob, type) and argspec == '()':
        # If fob has no argument, use default callable argspec.
        argspec = _default_callable_argspec

    lines = (textwrap.wrap(argspec, _MAX_COLS, subsequent_indent=_INDENT)
             if len(argspec) > _MAX_COLS else [argspec] if argspec else [])

    # Augment lines from docstring, if any, and join to get argspec.
    doc = inspect.getdoc(ob)
    if doc:
        for line in doc.split('\n', _MAX_LINES)[:_MAX_LINES]:
            line = line.strip()
            if not line:
                break
            if len(line) > _MAX_COLS:
                line = line[: _MAX_COLS - 3] + '...'
            lines.append(line)
    argspec = '\n'.join(lines)

    return argspec or _default_callable_argspec


if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_calltip', verbosity=2)
