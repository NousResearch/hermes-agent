"""Wrapper functions for Tcl/Tk.

Tkinter provides classes which allow the display, positioning and
control of widgets. Toplevel widgets are Tk and Toplevel. Other
widgets are Frame, Label, Entry, Text, Canvas, Button, Radiobutton,
Checkbutton, Scale, Listbox, Scrollbar, OptionMenu, Spinbox
LabelFrame and PanedWindow.

Properties of the widgets are specified with keyword arguments.
Keyword arguments have the same name as the corresponding resource
under Tk.

Widgets are positioned with one of the geometry managers Place, Pack
or Grid. These managers can be called with methods place, pack, grid
available in every Widget.

Actions are bound to events by resources (e.g. keyword argument
command) or with the method bind.

Example (Hello, World):
import tkinter
from tkinter.constants import *
tk = tkinter.Tk()
frame = tkinter.Frame(tk, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
label = tkinter.Label(frame, text="Hello, World")
label.pack(fill=X, expand=1)
button = tkinter.Button(frame,text="Exit",command=tk.destroy)
button.pack(side=BOTTOM)
tk.mainloop()
"""

import collections
import enum
import sys
import types

import _tkinter # If this fails your Python may not be configured for Tk
TclError = _tkinter.TclError
from tkinter.constants import *
import re

wantobjects = 1

TkVersion = float(_tkinter.TK_VERSION)
TclVersion = float(_tkinter.TCL_VERSION)

READABLE = _tkinter.READABLE
WRITABLE = _tkinter.WRITABLE
EXCEPTION = _tkinter.EXCEPTION


_magic_re = re.compile(r'([\\{}])')
_space_re = re.compile(r'([\s])', re.ASCII)


def _join(value):
    """Internal function."""
    return ' '.join(map(_stringify, value))


def _stringify(value):
    """Internal function."""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            value = _stringify(value[0])
            if _magic_re.search(value):
                value = '{%s}' % value
        else:
            value = '{%s}' % _join(value)
    else:
        value = str(value)
        if not value:
            value = '{}'
        elif _magic_re.search(value):
            # add '\' before special characters and spaces
            value = _magic_re.sub(r'\\\1', value)
            value = value.replace('\n', r'\n')
            value = _space_re.sub(r'\\\1', value)
            if value[0] == '"':
                value = '\\' + value
        elif value[0] == '"' or _space_re.search(value):
            value = '{%s}' % value
    return value


def _flatten(seq):
    """Internal function."""
    res = ()
    for item in seq:
        if isinstance(item, (tuple, list)):
            res = res + _flatten(item)
        elif item is not None:
            res = res + (item,)
    return res


try: _flatten = _tkinter._flatten
except AttributeError: pass


def _cnfmerge(cnfs):
    """Internal function."""
    if isinstance(cnfs, dict):
        return cnfs
    elif isinstance(cnfs, (type(None), str)):
        return cnfs
    else:
        cnf = {}
        for c in _flatten(cnfs):
            try:
                cnf.update(c)
            except (AttributeError, TypeError) as msg:
                print("_cnfmerge: fallback due to:", msg)
                for k, v in c.items():
                    cnf[k] = v
        return cnf


try: _cnfmerge = _tkinter._cnfmerge
except AttributeError: pass


def _splitdict(tk, v, cut_minus=True, conv=None):
    """Return a properly formatted dict built from Tcl list pairs.

    If cut_minus is True, the supposed '-' prefix will be removed from
    keys. If conv is specified, it is used to convert values.

    Tcl list is expected to contain an even number of elements.
    """
    t = tk.splitlist(v)
    if len(t) % 2:
        raise RuntimeError('Tcl list representing a dict is expected '
                           'to contain an even number of elements')
    it = iter(t)
    dict = {}
    for key, value in zip(it, it):
        key = str(key)
        if cut_minus and key[0] == '-':
            key = key[1:]
        if conv:
            value = conv(value)
        dict[key] = value
    return dict

class _VersionInfoType(collections.namedtuple('_VersionInfoType',
        ('major', 'minor', 'micro', 'releaselevel', 'serial'))):
    def __str__(self):
        if self.releaselevel == 'final':
            return f'{self.major}.{self.minor}.{self.micro}'
        else:
            return f'{self.major}.{self.minor}{self.releaselevel[0]}{self.serial}'

def _parse_version(version):
    import re
    m = re.fullmatch(r'(\d+)\.(\d+)([ab.])(\d+)', version)
    major, minor, releaselevel, serial = m.groups()
    major, minor, serial = int(major), int(minor), int(serial)
    if releaselevel == '.':
        micro = serial
        serial = 0
        releaselevel = 'final'
    else:
        micro = 0
        releaselevel = {'a': 'alpha', 'b': 'beta'}[releaselevel]
    return _VersionInfoType(major, minor, micro, releaselevel, serial)


@enum._simple_enum(enum.StrEnum)
class EventType:
    KeyPress = '2'
    Key = KeyPress
    KeyRelease = '3'
    ButtonPress = '4'
    Button = ButtonPress
    ButtonRelease = '5'
    Motion = '6'
    Enter = '7'
    Leave = '8'
    FocusIn = '9'
    FocusOut = '10'
    Keymap = '11'           # undocumented
    Expose = '12'
    GraphicsExpose = '13'   # undocumented
    NoExpose = '14'         # undocumented
    Visibility = '15'
    Create = '16'
    Destroy = '17'
    Unmap = '18'
    Map = '19'
    MapRequest = '20'
    Reparent = '21'
    Configure = '22'
    ConfigureRequest = '23'
    Gravity = '24'
    ResizeRequest = '25'
    Circulate = '26'
    CirculateRequest = '27'
    Property = '28'
    SelectionClear = '29'   # undocumented
    SelectionRequest = '30' # undocumented
    Selection = '31'        # undocumented
    Colormap = '32'
    ClientMessage = '33'    # undocumented
    Mapping = '34'          # undocumented
    VirtualEvent = '35'     # undocumented
    Activate = '36'
    Deactivate = '37'
    MouseWheel = '38'


class Event:
    """Container for the properties of an event.

    Instances of this type are generated if one of the following events occurs:

    KeyPress, KeyRelease - for keyboard events
    ButtonPress, ButtonRelease, Motion, Enter, Leave, MouseWheel - for mouse events
    Visibility, Unmap, Map, Expose, FocusIn, FocusOut, Circulate,
    Colormap, Gravity, Reparent, Property, Destroy, Activate,
    Deactivate - for window events.

    If a callback function for one of these events is registered
    using bind, bind_all, bind_class, or tag_bind, the callback is
    called with an Event as first argument. It will have the
    following attributes (in braces are the event types for which
    the attribute is valid):

        serial - serial number of event
    num - mouse button pressed (ButtonPress, ButtonRelease)
    focus - whether the window has the focus (Enter, Leave)
    height - height of the exposed window (Configure, Expose)
    width - width of the exposed window (Configure, Expose)
    keycode - keycode of the pressed key (KeyPress, KeyRelease)
    state - state of the event as a number (ButtonPress, ButtonRelease,
                            Enter, KeyPress, KeyRelease,
                            Leave, Motion)
    state - state as a string (Visibility)
    time - when the event occurred
    x - x-position of the mouse
    y - y-position of the mouse
    x_root - x-position of the mouse on the screen
             (ButtonPress, ButtonRelease, KeyPress, KeyRelease, Motion)
    y_root - y-position of the mouse on the screen
             (ButtonPress, ButtonRelease, KeyPress, KeyRelease, Motion)
    char - pressed character (KeyPress, KeyRelease)
    send_event - see X/Windows documentation
    keysym - keysym of the event as a string (KeyPress, KeyRelease)
    keysym_num - keysym of the event as a number (KeyPress, KeyRelease)
    type - type of the event as a number
    widget - widget in which the event occurred
    delta - delta of wheel movement (MouseWheel)
    """

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if v != '??'}
        if not self.char:
            del attrs['char']
        elif self.char != '??':
            attrs['char'] = repr(self.char)
        if not getattr(self, 'send_event', True):
            del attrs['send_event']
        if self.state == 0:
            del attrs['state']
        elif isinstance(self.state, int):
            state = self.state
            mods = ('Shift', 'Lock', 'Control',
                    'Mod1', 'Mod2', 'Mod3', 'Mod4', 'Mod5',
                    'Button1', 'Button2', 'Button3', 'Button4', 'Button5')
            s = []
            for i, n in enumerate(mods):
                if state & (1 << i):
                    s.append(n)
            state = state & ~((1<< len(mods)) - 1)
            if state or not s:
                s.append(hex(state))
            attrs['state'] = '|'.join(s)
        if self.delta == 0:
            del attrs['delta']
        # widget usually is known
        # serial and time are not very interesting
        # keysym_num duplicates keysym
        # x_root and y_root mostly duplicate x and y
        keys = ('send_event',
                'state', 'keysym', 'keycode', 'char',
                'num', 'delta', 'focus',
                'x', 'y', 'width', 'height')
        return '<%s event%s>' % (
            getattr(self.type, 'name', self.type),
            ''.join(' %s=%s' % (k, attrs[k]) for k in keys if k in attrs)
        )


_support_default_root = True
_default_root = None


def NoDefaultRoot():
    """Inhibit setting of default root window.

    Call this function to inhibit that the first instance of
    Tk is used for windows without an explicit parent window.
    """
    global _support_default_root, _default_root
    _support_default_root = False
    # Delete, so any use of _default_root will immediately raise an exception.
    # Rebind before deletion, so repeated calls will not fail.
    _default_root = None
    del _default_root


def _get_default_root(what=None):
    if not _support_default_root:
        raise RuntimeError("No master specified and tkinter is "
                           "configured to not support default root")
    if _default_root is None:
        if what:
            raise RuntimeError(f"Too early to {what}: no default root window")
        root = Tk()
        assert _default_root is root
    return _default_root


def _get_temp_root():
    global _support_default_root
    if not _support_default_root:
        raise RuntimeError("No master specified and tkinter is "
                           "configured to not support default root")
    root = _default_root
    if root is None:
        assert _support_default_root
        _support_default_root = False
        root = Tk()
        _support_default_root = True
        assert _default_root is None
        root.withdraw()
        root._temporary = True
    return root


def _destroy_temp_root(master):
    if getattr(master, '_temporary', False):
        try:
            master.destroy()
        except TclError:
            pass


def _tkerror(err):
    """Internal function."""
    pass


def _exit(code=0):
    """Internal function. Calling it will raise the exception SystemExit."""
    try:
        code = int(code)
    except ValueError:
        pass
    raise SystemExit(code)


_varnum = 0


class Variable:
    """Class to define value holders for e.g. buttons.

    Subclasses StringVar, IntVar, DoubleVar, BooleanVar are specializations
    that constrain the type of the value returned from get()."""
    _default = ""
    _tk = None
    _tclCommands = None

    def __init__(self, master=None, value=None, name=None):
        """Construct a variable

        MASTER can be given as master widget.
        VALUE is an optional value (defaults to "")
        NAME is an optional Tcl name (defaults to PY_VARnum).

        If NAME matches an existing variable and VALUE is omitted
        then the existing value is retained.
        """
        # check for type of NAME parameter to override weird error message
        # raised from Modules/_tkinter.c:SetVar like:
        # TypeError: setvar() takes exactly 3 arguments (2 given)
        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string")
        global _varnum
        if master is None:
            master = _get_default_root('create variable')
        self._root = master._root()
        self._tk = master.tk
        if name:
            self._name = name
        else:
            self._name = 'PY_VAR' + repr(_varnum)
            _varnum += 1
        if value is not None:
            self.initialize(value)
        elif not self._tk.getboolean(self._tk.call("info", "exists", self._name)):
            self.initialize(self._default)

    def __del__(self):
        """Unset the variable in Tcl."""
        if self._tk is None:
            return
        if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
            self._tk.globalunsetvar(self._name)
        if self._tclCommands is not None:
            for name in self._tclCommands:
                #print '- Tkinter: deleted command', name
                self._tk.deletecommand(name)
            self._tclCommands = None

    def __str__(self):
        """Return the name of the variable in Tcl."""
        return self._name

    def set(self, value):
        """Set the variable to VALUE."""
        return self._tk.globalsetvar(self._name, value)

    initialize = set

    def get(self):
        """Return value of variable."""
        return self._tk.globalgetvar(self._name)

    def _register(self, callback):
        f = CallWrapper(callback, None, self._root).__call__
        cbname = repr(id(f))
        try:
            callback = callback.__func__
        except AttributeError:
            pass
        try:
            cbname = cbname + callback.__name__
        except AttributeError:
            pass
        self._tk.createcommand(cbname, f)
        if self._tclCommands is None:
            self._tclCommands = []
        self._tclCommands.append(cbname)
        return cbname

    def trace_add(self, mode, callback):
        """Define a trace callback for the variable.

        Mode is one of "read", "write", "unset", or a list or tuple of
        such strings.
        Callback must be a function which is called when the variable is
        read, written or unset.

        Return the name of the callback.
        """
        cbname = self._register(callback)
        self._tk.call('trace', 'add', 'variable',
                      self._name, mode, (cbname,))
        return cbname

    def trace_remove(self, mode, cbname):
        """Delete the trace callback for a variable.

        Mode is one of "read", "write", "unset" or a list or tuple of
        such strings.  Must be same as were specified in trace_add().
        cbname is the name of the callback returned from trace_add().
        """
        self._tk.call('trace', 'remove', 'variable',
                      self._name, mode, cbname)
        for m, ca in self.trace_info():
            if self._tk.splitlist(ca)[0] == cbname:
                break
        else:
            self._tk.deletecommand(cbname)
            try:
                self._tclCommands.remove(cbname)
            except ValueError:
                pass

    def trace_info(self):
        """Return all trace callback information."""
        splitlist = self._tk.splitlist
        return [(splitlist(k), v) for k, v in map(splitlist,
            splitlist(self._tk.call('trace', 'info', 'variable', self._name)))]

    def trace_variable(self, mode, callback):
        """Define a trace callback for the variable.

        MODE is one of "r", "w", "u" for read, write, undefine.
        CALLBACK must be a function which is called when
        the variable is read, written or undefined.

        Return the name of the callback.

        This deprecated method wraps a deprecated Tcl method that will
        likely be removed in the future.  Use trace_add() instead.
        """
        # TODO: Add deprecation warning
        cbname = self._register(callback)
        self._tk.call("trace", "variable", self._name, mode, cbname)
        return cbname

    trace = trace_variable

    def trace_vdelete(self, mode, cbname):
        """Delete the trace callback for a variable.

        MODE is one of "r", "w", "u" for read, write, undefine.
        CBNAME is the name of the callback returned from trace_variable or trace.

        This deprecated method wraps a deprecated Tcl method that will
        likely be removed in the future.  Use trace_remove() instead.
        """
        # TODO: Add deprecation warning
        self._tk.call("trace", "vdelete", self._name, mode, cbname)
        cbname = self._tk.splitlist(cbname)[0]
        for m, ca in self.trace_info():
            if self._tk.splitlist(ca)[0] == cbname:
                break
        else:
            self._tk.deletecommand(cbname)
            try:
                self._tclCommands.remove(cbname)
            except ValueError:
                pass

    def trace_vinfo(self):
        """Return all trace callback information.

        This deprecated method wraps a deprecated Tcl method that will
        likely be removed in the future.  Use trace_info() instead.
        """
        # TODO: Add deprecation warning
        return [self._tk.splitlist(x) for x in self._tk.splitlist(
            self._tk.call("trace", "vinfo", self._name))]

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return NotImplemented
        return (self._name == other._name
                and self.__class__.__name__ == other.__class__.__name__
                and self._tk == other._tk)


class StringVar(Variable):
    """Value holder for strings variables."""
    _default = ""

    def __init__(self, master=None, value=None, name=None):
        """Construct a string variable.

        MASTER can be given as master widget.
        VALUE is an optional value (defaults to "")
        NAME is an optional Tcl name (defaults to PY_VARnum).

        If NAME matches an existing variable and VALUE is omitted
        then the existing value is retained.
        """
        Variable.__init__(self, master, value, name)

    def get(self):
        """Return value of variable as string."""
        value = self._tk.globalgetvar(self._name)
        if isinstance(value, str):
            return value
        return str(value)


class IntVar(Variable):
    """Value holder for integer variables."""
    _default = 0

    def __init__(self, master=None, value=None, name=None):
        """Construct an integer variable.

        MASTER can be given as master widget.
        VALUE is an optional value (defaults to 0)
        NAME is an optional Tcl name (defaults to PY_VARnum).

        If NAME matches an existing variable and VALUE is omitted
        then the existing value is retained.
        """
        Variable.__init__(self, master, value, name)

    def get(self):
        """Return the value of the variable as an integer."""
        value = self._tk.globalgetvar(self._name)
        try:
            return self._tk.getint(value)
        except (TypeError, TclError):
            return int(self._tk.getdouble(value))


class DoubleVar(Variable):
    """Value holder for float variables."""
    _default = 0.0

    def __init__(self, master=None, value=None, name=None):
        """Construct a float variable.

        MASTER can be given as master widget.
        VALUE is an optional value (defaults to 0.0)
        NAME is an optional Tcl name (defaults to PY_VARnum).

        If NAME matches an existing variable and VALUE is omitted
        then the existing value is retained.
        """
        Variable.__init__(self, master, value, name)

    def get(self):
        """Return the value of the variable as a float."""
        return self._tk.getdouble(self._tk.globalgetvar(self._name))


class BooleanVar(Variable):
    """Value holder for boolean variables."""
    _default = False

    def __init__(self, master=None, value=None, name=None):
        """Construct a boolean variable.

        MASTER can be given as master widget.
        VALUE is an optional value (defaults to False)
        NAME is an optional Tcl name (defaults to PY_VARnum).

        If NAME matches an existing variable and VALUE is omitted
        then the existing value is retained.
        """
        Variable.__init__(self, master, value, name)

    def set(self, value):
        """Set the variable to VALUE."""
        return self._tk.globalsetvar(self._name, self._tk.getboolean(value))

    initialize = set

    def get(self):
        """Return the value of the variable as a bool."""
        try:
            return self._tk.getboolean(self._tk.globalgetvar(self._name))
        except TclError:
            raise ValueError("invalid literal for getboolean()")


def mainloop(n=0):
    """Run the main loop of Tcl."""
    _get_default_root('run the main loop').tk.mainloop(n)


getint = int

getdouble = float


def getboolean(s):
    """Convert Tcl object to True or False."""
    try:
        return _get_default_root('use getboolean()').tk.getboolean(s)
    except TclError:
        raise ValueError("invalid literal for getboolean()")


# Methods defined on both toplevel and interior widgets

class Misc:
    """Internal class.

    Base class which defines methods common for interior widgets."""

    # used for generating child widget names
    _last_child_ids = None

    # XXX font command?
    _tclCommands = None

    def destroy(self):
        """Internal function.

        Delete all Tcl commands created for
        this widget in the Tcl interpreter."""
        if self._tclCommands is not None:
            for name in self._tclCommands:
                #print '- Tkinter: deleted command', name
                self.tk.deletecommand(name)
            self._tclCommands = None

    def deletecommand(self, name):
        """Internal function.

        Delete the Tcl command provided in NAME."""
        #print '- Tkinter: deleted command', name
        self.tk.deletecommand(name)
        try:
            self._tclCommands.remove(name)
        except ValueError:
            pass

    def tk_strictMotif(self, boolean=None):
        """Set Tcl internal variable, whether the look and feel
        should adhere to Motif.

        A parameter of 1 means adhere to Motif (e.g. no color
        change if mouse passes over slider).
        Returns the set value."""
        return self.tk.getboolean(self.tk.call(
            'set', 'tk_strictMotif', boolean))

    def tk_bisque(self):
        """Change the color scheme to light brown as used in Tk 3.6 and before."""
        self.tk.call('tk_bisque')

    def tk_setPalette(self, *args, **kw):
        """Set a new color scheme for all widget elements.

        A single color as argument will cause that all colors of Tk
        widget elements are derived from this.
        Alternatively several keyword parameters and its associated
        colors can be given. The following keywords are valid:
        activeBackground, foreground, selectColor,
        activeForeground, highlightBackground, selectBackground,
        background, highlightColor, selectForeground,
        disabledForeground, insertBackground, troughColor."""
        self.tk.call(('tk_setPalette',)
              + _flatten(args) + _flatten(list(kw.items())))

    def wait_variable(self, name='PY_VAR'):
        """Wait until the variable is modified.

        A parameter of type IntVar, StringVar, DoubleVar or
        BooleanVar must be given."""
        self.tk.call('tkwait', 'variable', name)
    waitvar = wait_variable # XXX b/w compat

    def wait_window(self, window=None):
        """Wait until a WIDGET is destroyed.

        If no parameter is given self is used."""
        if window is None:
            window = self
        self.tk.call('tkwait', 'window', window._w)

    def wait_visibility(self, window=None):
        """Wait until the visibility of a WIDGET changes
        (e.g. it appears).

        If no parameter is given self is used."""
        if window is None:
            window = self
        self.tk.call('tkwait', 'visibility', window._w)

    def setvar(self, name='PY_VAR', value='1'):
        """Set Tcl variable NAME to VALUE."""
        self.tk.setvar(name, value)

    def getvar(self, name='PY_VAR'):
        """Return value of Tcl variable NAME."""
        return self.tk.getvar(name)

    def getint(self, s):
        try:
            return self.tk.getint(s)
        except TclError as exc:
            raise ValueError(str(exc))

    def getdouble(self, s):
        try:
            return self.tk.getdouble(s)
        except TclError as exc:
            raise ValueError(str(exc))

    def getboolean(self, s):
        """Return a boolean value for Tcl boolean values true and false given as parameter."""
        try:
            return self.tk.getboolean(s)
        except TclError:
            raise ValueError("invalid literal for getboolean()")

    def focus_set(self):
        """Direct input focus to this widget.

        If the application currently does not have the focus
        this widget will get the focus if the application gets
        the focus through the window manager."""
        self.tk.call('focus', self._w)
    focus = focus_set # XXX b/w compat?

    def focus_force(self):
        """Direct input focus to this widget even if the
        application does not have the focus. Use with
        caution!"""
        self.tk.call('focus', '-force', self._w)

    def focus_get(self):
        """Return the widget which has currently the focus in the
        application.

        Use focus_displayof to allow working with several
        displays. Return None if application does not have
        the focus."""
        name = self.tk.call('focus')
        if name == 'none' or not name: return None
        return self._nametowidget(name)

    def focus_displayof(self):
        """Return the widget which has currently the focus on the
        display where this widget is located.

        Return None if the application does not have the focus."""
        name = self.tk.call('focus', '-displayof', self._w)
        if name == 'none' or not name: return None
        return self._nametowidget(name)

    def focus_lastfor(self):
        """Return the widget which would have the focus if top level
        for this widget gets the focus from the window manager."""
        name = self.tk.call('focus', '-lastfor', self._w)
        if name == 'none' or not name: return None
        return self._nametowidget(name)

    def tk_focusFollowsMouse(self):
        """The widget under mouse will get automatically focus. Can not
        be disabled easily."""
        self.tk.call('tk_focusFollowsMouse')

    def tk_focusNext(self):
        """Return the next widget in the focus order which follows
        widget which has currently the focus.

        The focus order first goes to the next child, then to
        the children of the child recursively and then to the
        next sibling which is higher in the stacking order.  A
        widget is omitted if it has the takefocus resource set
        to 0."""
        name = self.tk.call('tk_focusNext', self._w)
        if not name: return None
        return self._nametowidget(name)

    def tk_focusPrev(self):
        """Return previous widget in the focus order. See tk_focusNext for details."""
        name = self.tk.call('tk_focusPrev', self._w)
        if not name: return None
        return self._nametowidget(name)

    def after(self, ms, func=None, *args):
        """Call function once after given time.

        MS specifies the time in milliseconds. FUNC gives the
        function which shall be called. Additional parameters
        are given as parameters to the function call.  Return
        identifier to cancel scheduling with after_cancel."""
        if func is None:
            # I'd rather use time.sleep(ms*0.001)
            self.tk.call('after', ms)
            return None
        else:
            def callit():
                try:
                    func(*args)
                finally:
                    try:
                        self.deletecommand(name)
                    except TclError:
                        pass
            try:
                callit.__name__ = func.__name__
            except AttributeError:
                # Required for callable classes (bpo-44404)
                callit.__name__ = type(func).__name__
            name = self._register(callit)
            return self.tk.call('after', ms, name)

    def after_idle(self, func, *args):
        """Call FUNC once if the Tcl main loop has no event to
        process.

        Return an identifier to cancel the scheduling with
        after_cancel."""
        return self.after('idle', func, *args)

    def after_cancel(self, id):
        """Cancel scheduling of function identified with ID.

        Identifier returned by after or after_idle must be
        given as first parameter.
        """
        if not id:
            raise ValueError('id must be a valid identifier returned from '
                             'after or after_idle')
        try:
            data = self.tk.call('after', 'info', id)
            script = self.tk.splitlist(data)[0]
            self.deletecommand(script)
        except TclError:
            pass
        self.tk.call('after', 'cancel', id)

    def bell(self, displayof=0):
        """Ring a display's bell."""
        self.tk.call(('bell',) + self._displayof(displayof))

    # Clipboard handling:
    def clipboard_get(self, **kw):
        """Retrieve data from the clipboard on window's display.

        The window keyword defaults to the root window of the Tkinter
        application.

        The type keyword specifies the form in which the data is
        to be returned and should be an atom name such as STRING
        or FILE_NAME.  Type defaults to STRING, except on X11, where the default
        is to try UTF8_STRING and fall back to STRING.

        This command is equivalent to:

        selection_get(CLIPBOARD)
        """
        if 'type' not in kw and self._windowingsystem == 'x11':
            try:
                kw['type'] = 'UTF8_STRING'
                return self.tk.call(('clipboard', 'get') + self._options(kw))
            except TclError:
                del kw['type']
        return self.tk.call(('clipboard', 'get') + self._options(kw))

    def clipboard_clear(self, **kw):
        """Clear the data in the Tk clipboard.

        A widget specified for the optional displayof keyword
        argument specifies the target display."""
        if 'displayof' not in kw: kw['displayof'] = self._w
        self.tk.call(('clipboard', 'clear') + self._options(kw))

    def clipboard_append(self, string, **kw):
        """Append STRING to the Tk clipboard.

        A widget specified at the optional displayof keyword
        argument specifies the target display. The clipboard
        can be retrieved with selection_get."""
        if 'displayof' not in kw: kw['displayof'] = self._w
        self.tk.call(('clipboard', 'append') + self._options(kw)
              + ('--', string))
    # XXX grab current w/o window argument

    def grab_current(self):
        """Return widget which has currently the grab in this application
        or None."""
        name = self.tk.call('grab', 'current', self._w)
        if not name: return None
        return self._nametowidget(name)

    def grab_release(self):
        """Release grab for this widget if currently set."""
        self.tk.call('grab', 'release', self._w)

    def grab_set(self):
        """Set grab for this widget.

        A grab directs all events to this and descendant
        widgets in the application."""
        self.tk.call('grab', 'set', self._w)

    def grab_set_global(self):
        """Set global grab for this widget.

        A global grab directs all events to this and
        descendant widgets on the display. Use with caution -
        other applications do not get events anymore."""
        self.tk.call('grab', 'set', '-global', self._w)

    def grab_status(self):
        """Return None, "local" or "global" if this widget has
        no, a local or a global grab."""
        status = self.tk.call('grab', 'status', self._w)
        if status == 'none': status = None
        return status

    def option_add(self, pattern, value, priority = None):
        """Set a VALUE (second parameter) for an option
        PATTERN (first parameter).

        An optional third parameter gives the numeric priority
        (defaults to 80)."""
        self.tk.call('option', 'add', pattern, value, priority)

    def option_clear(self):
        """Clear the option database.

        It will be reloaded if option_add is called."""
        self.tk.call('option', 'clear')

    def option_get(self, name, className):
        """Return the value for an option NAME for this widget
        with CLASSNAME.

        Values with higher priority override lower values."""
        return self.tk.call('option', 'get', self._w, name, className)

    def option_readfile(self, fileName, priority = None):
        """Read file FILENAME into the option database.

        An optional second parameter gives the numeric
        priority."""
        self.tk.call('option', 'readfile', fileName, priority)

    def selection_clear(self, **kw):
        """Clear the current X selection."""
        if 'displayof' not in kw: kw['displayof'] = self._w
        self.tk.call(('selection', 'clear') + self._options(kw))

    def selection_get(self, **kw):
        """Return the contents of the current X selection.

        A keyword parameter selection specifies the name of
        the selection and defaults to PRIMARY.  A keyword
        parameter displayof specifies a widget on the display
        to use. A keyword parameter type specifies the form of data to be
        fetched, defaulting to STRING except on X11, where UTF8_STRING is tried
        before STRING."""
        if 'displayof' not in kw: kw['displayof'] = self._w
        if 'type' not in kw and self._windowingsystem == 'x11':
            try:
                kw['type'] = 'UTF8_STRING'
                return self.tk.call(('selection', 'get') + self._options(kw))
            except TclError:
                del kw['type']
        return self.tk.call(('selection', 'get') + self._options(kw))

    def selection_handle(self, command, **kw):
        """Specify a function COMMAND to call if the X
        selection owned by this widget is queried by another
        application.

        This function must return the contents of the
        selection. The function will be called with the
        arguments OFFSET and LENGTH which allows the chunking
        of very long selections. The following keyword
        parameters can be provided:
        selection - name of the selection (default PRIMARY),
        type - type of the selection (e.g. STRING, FILE_NAME)."""
        name = self._register(command)
        self.tk.call(('selection', 'handle') + self._options(kw)
              + (self._w, name))

    def selection_own(self, **kw):
        """Become owner of X selection.

        A keyword parameter selection specifies the name of
        the selection (default PRIMARY)."""
        self.tk.call(('selection', 'own') +
                 self._options(kw) + (self._w,))

    def selection_own_get(self, **kw):
        """Return owner of X selection.

        The following keyword parameter can
        be provided:
        selection - name of the selection (default PRIMARY),
        type - type of the selection (e.g. STRING, FILE_NAME)."""
        if 'displayof' not in kw: kw['displayof'] = self._w
        name = self.tk.call(('selection', 'own') + self._options(kw))
        if not name: return None
        return self._nametowidget(name)

    def send(self, interp, cmd, *args):
        """Send Tcl command CMD to different interpreter INTERP to be executed."""
        return self.tk.call(('send', interp, cmd) + args)

    def lower(self, belowThis=None):
        """Lower this widget in the stacking order."""
        self.tk.call('lower', self._w, belowThis)

    def tkraise(self, aboveThis=None):
        """Raise this widget in the stacking order."""
        self.tk.call('raise', self._w, aboveThis)

    lift = tkraise

    def info_patchlevel(self):
        """Returns the exact version of the Tcl library."""
        patchlevel = self.tk.call('info', 'patchlevel')
        return _parse_version(patchlevel)

    def winfo_atom(self, name, displayof=0):
        """Return integer which represents atom NAME."""
        args = ('winfo', 'atom') + self._displayof(displayof) + (name,)
        return self.tk.getint(self.tk.call(args))

    def winfo_atomname(self, id, displayof=0):
        """Return name of atom with identifier ID."""
        args = ('winfo', 'atomname') \
               + self._displayof(displayof) + (id,)
        return self.tk.call(args)

    def winfo_cells(self):
        """Return number of cells in the colormap for this widget."""
        return self.tk.getint(
            self.tk.call('winfo', 'cells', self._w))

    def winfo_children(self):
        """Return a list of all widgets which are children of this widget."""
        result = []
        for child in self.tk.splitlist(
            self.tk.call('winfo', 'children', self._w)):
            try:
                # Tcl sometimes returns extra windows, e.g. for
                # menus; those need to be skipped
                result.append(self._nametowidget(child))
            except KeyError:
                pass
        return result

    def winfo_class(self):
        """Return window class name of this widget."""
        return self.tk.call('winfo', 'class', self._w)

    def winfo_colormapfull(self):
        """Return True if at the last color request the colormap was full."""
        return self.tk.getboolean(
            self.tk.call('winfo', 'colormapfull', self._w))

    def winfo_containing(self, rootX, rootY, displayof=0):
        """Return the widget which is at the root coordinates ROOTX, ROOTY."""
        args = ('winfo', 'containing') \
               + self._displayof(displayof) + (rootX, rootY)
        name = self.tk.call(args)
        if not name: return None
        return self._nametowidget(name)

    def winfo_depth(self):
        """Return the number of bits per pixel."""
        return self.tk.getint(self.tk.call('winfo', 'depth', self._w))

    def winfo_exists(self):
        """Return true if this widget exists."""
        return self.tk.getint(
            self.tk.call('winfo', 'exists', self._w))

    def winfo_fpixels(self, number):
        """Return the number of pixels for the given distance NUMBER
        (e.g. "3c") as float."""
        return self.tk.getdouble(self.tk.call(
            'winfo', 'fpixels', self._w, number))

    def winfo_geometry(self):
        """Return geometry string for this widget in the form "widthxheight+X+Y"."""
        return self.tk.call('winfo', 'geometry', self._w)

    def winfo_height(self):
        """Return height of this widget."""
        return self.tk.getint(
            self.tk.call('winfo', 'height', self._w))

    def winfo_id(self):
        """Return identifier ID for this widget."""
        return int(self.tk.call('winfo', 'id', self._w), 0)

    def winfo_interps(self, displayof=0):
        """Return the name of all Tcl interpreters for this display."""
        args = ('winfo', 'interps') + self._displayof(displayof)
        return self.tk.splitlist(self.tk.call(args))

    def winfo_ismapped(self):
        """Return true if this widget is mapped."""
        return self.tk.getint(
            self.tk.call('winfo', 'ismapped', self._w))

    def winfo_manager(self):
        """Return the window manager name for this widget."""
        return self.tk.call('winfo', 'manager', self._w)

    def winfo_name(self):
        """Return the name of this widget."""
        return self.tk.call('winfo', 'name', self._w)

    def winfo_parent(self):
        """Return the name of the parent of this widget."""
        return self.tk.call('winfo', 'parent', self._w)

    def winfo_pathname(self, id, displayof=0):
        """Return the pathname of the widget given by ID."""
        if isinstance(id, int):
            id = hex(id)
        args = ('winfo', 'pathname') \
               + self._displayof(displayof) + (id,)
        return self.tk.call(args)

    def winfo_pixels(self, number):
        """Rounded integer value of winfo_fpixels."""
        return self.tk.getint(
            self.tk.call('winfo', 'pixels', self._w, number))

    def winfo_pointerx(self):
        """Return the x coordinate of the pointer on the root window."""
        return self.tk.getint(
            self.tk.call('winfo', 'pointerx', self._w))

    def winfo_pointerxy(self):
        """Return a tuple of x and y coordinates of the pointer on the root window."""
        return self._getints(
            self.tk.call('winfo', 'pointerxy', self._w))

    def winfo_pointery(self):
        """Return the y coordinate of the pointer on the root window."""
        return self.tk.getint(
            self.tk.call('winfo', 'pointery', self._w))

    def winfo_reqheight(self):
        """Return requested height of this widget."""
        return self.tk.getint(
            self.tk.call('winfo', 'reqheight', self._w))

    def winfo_reqwidth(self):
        """Return requested width of this widget."""
        return self.tk.getint(
            self.tk.call('winfo', 'reqwidth', self._w))

    def winfo_rgb(self, color):
        """Return a tuple of integer RGB values in range(65536) for color in this widget."""
        return self._getints(
            self.tk.call('winfo', 'rgb', self._w, color))

    def winfo_rootx(self):
        """Return x coordinate of upper left corner of this widget on the
        root window."""
        return self.tk.getint(
            self.tk.call('winfo', 'rootx', self._w))

    def winfo_rooty(self):
        """Return y coordinate of upper left corner of this widget on the
        root window."""
        return self.tk.getint(
            self.tk.call('winfo', 'rooty', self._w))

    def winfo_screen(self):
        """Return the screen name of this widget."""
        return self.tk.call('winfo', 'screen', self._w)

    def winfo_screencells(self):
        """Return the number of the cells in the colormap of the screen
        of this widget."""
        return self.tk.getint(
            self.tk.call('winfo', 'screencells', self._w))

    def winfo_screendepth(self):
        """Return the number of bits per pixel of the root window of the
        screen of this widget."""
        return self.tk.getint(
            self.tk.call('winfo', 'screendepth', self._w))

    def winfo_screenheight(self):
        """Return the number of pixels of the height of the screen of this widget
        in pixel."""
        return self.tk.getint(
            self.tk.call('winfo', 'screenheight', self._w))

    def winfo_screenmmheight(self):
        """Return the number of pixels of the height of the screen of
        this widget in mm."""
        return self.tk.getint(
            self.tk.call('winfo', 'screenmmheight', self._w))

    def winfo_screenmmwidth(self):
        """Return the number of pixels of the width of the screen of
        this widget in mm."""
        return self.tk.getint(
            self.tk.call('winfo', 'screenmmwidth', self._w))

    def winfo_screenvisual(self):
        """Return one of the strings directcolor, grayscale, pseudocolor,
        staticcolor, staticgray, or truecolor for the default
        colormodel of this screen."""
        return self.tk.call('winfo', 'screenvisual', self._w)

    def winfo_screenwidth(self):
        """Return the number of pixels of the width of the screen of
        this widget in pixel."""
        return self.tk.getint(
            self.tk.call('winfo', 'screenwidth', self._w))

    def winfo_server(self):
        """Return information of the X-Server of the screen of this widget in
        the form "XmajorRminor vendor vendorVersion"."""
        return self.tk.call('winfo', 'server', self._w)

    def winfo_toplevel(self):
        """Return the toplevel widget of this widget."""
        return self._nametowidget(self.tk.call(
            'winfo', 'toplevel', self._w))

    def winfo_viewable(self):
        """Return true if the widget and all its higher ancestors are mapped."""
        return self.tk.getint(
            self.tk.call('winfo', 'viewable', self._w))

    def winfo_visual(self):
        """Return one of the strings directcolor, grayscale, pseudocolor,
        staticcolor, staticgray, or truecolor for the
        colormodel of this widget."""
        return self.tk.call('winfo', 'visual', self._w)

    def winfo_visualid(self):
        """Return the X identifier for the visual for this widget."""
        return self.tk.call('winfo', 'visualid', self._w)

    def winfo_visualsavailable(self, includeids=False):
        """Return a list of all visuals available for the screen
        of this widget.

        Each item in the list consists of a visual name (see winfo_visual), a
        depth and if includeids is true is given also the X identifier."""
        data = self.tk.call('winfo', 'visualsavailable', self._w,
                            'includeids' if includeids else None)
        data = [self.tk.splitlist(x) for x in self.tk.splitlist(data)]
        return [self.__winfo_parseitem(x) for x in data]

    def __winfo_parseitem(self, t):
        """Internal function."""
        return t[:1] + tuple(map(self.__winfo_getint, t[1:]))

    def __winfo_getint(self, x):
        """Internal function."""
        return int(x, 0)

    def winfo_vrootheight(self):
        """Return the height of the virtual root window associated with this
        widget in pixels. If there is no virtual root window return the
        height of the screen."""
        return self.tk.getint(
            self.tk.call('winfo', 'vrootheight', self._w))

    def winfo_vrootwidth(self):
        """Return the width of the virtual root window associated with this
        widget in pixel. If there is no virtual root window return the
        width of the screen."""
        return self.tk.getint(
            self.tk.call('winfo', 'vrootwidth', self._w))

    def winfo_vrootx(self):
        """Return the x offset of the virtual root relative to the root
        window of the screen of this widget."""
        return self.tk.getint(
            self.tk.call('winfo', 'vrootx', self._w))

    def winfo_vrooty(self):
        """Return the y offset of the virtual root relative to the root
        window of the screen of this widget."""
        return self.tk.getint(
            self.tk.call('winfo', 'vrooty', self._w))

    def winfo_width(self):
        """Return the width of this widget."""
        return self.tk.getint(
            self.tk.call('winfo', 'width', self._w))

    def winfo_x(self):
        """Return the x coordinate of the upper left corner of this widget
        in the parent."""
        return self.tk.getint(
            self.tk.call('winfo', 'x', self._w))

    def winfo_y(self):
        """Return the y coordinate of the upper left corner of this widget
        in the parent."""
        return self.tk.getint(
            self.tk.call('winfo', 'y', self._w))

    def update(self):
        """Enter event loop until all pending events have been processed by Tcl."""
        self.tk.call('update')

    def update_idletasks(self):
        """Enter event loop until all idle callbacks have been called. This
        will update the display of windows but not process events caused by
        the user."""
        self.tk.call('update', 'idletasks')

    def bindtags(self, tagList=None):
        """Set or get the list of bindtags for this widget.

        With no argument return the list of all bindtags associated with
        this widget. With a list of strings as argument the bindtags are
        set to this list. The bindtags determine in which order events are
        processed (see bind)."""
        if tagList is None:
            return self.tk.splitlist(
                self.tk.call('bindtags', self._w))
        else:
            self.tk.call('bindtags', self._w, tagList)

    def _bind(self, what, sequence, func, add, needcleanup=1):
        """Internal function."""
        if isinstance(func, str):
            self.tk.call(what + (sequence, func))
        elif func:
            funcid = self._register(func, self._substitute,
                        needcleanup)
            cmd = ('%sif {"[%s %s]" == "break"} break\n'
                   %
                   (add and '+' or '',
                funcid, self._subst_format_str))
            self.tk.call(what + (sequence, cmd))
            return funcid
        elif sequence:
            return self.tk.call(what + (sequence,))
        else:
            return self.tk.splitlist(self.tk.call(what))

    def bind(self, sequence=None, func=None, add=None):
        """Bind to this widget at event SEQUENCE a call to function FUNC.

        SEQUENCE is a string of concatenated event
        patterns. An event pattern is of the form
        <MODIFIER-MODIFIER-TYPE-DETAIL> where MODIFIER is one
        of Control, Mod2, M2, Shift, Mod3, M3, Lock, Mod4, M4,
        Button1, B1, Mod5, M5 Button2, B2, Meta, M, Button3,
        B3, Alt, Button4, B4, Double, Button5, B5 Triple,
        Mod1, M1. TYPE is one of Activate, Enter, Map,
        ButtonPress, Button, Expose, Motion, ButtonRelease
        FocusIn, MouseWheel, Circulate, FocusOut, Property,
        Colormap, Gravity Reparent, Configure, KeyPress, Key,
        Unmap, Deactivate, KeyRelease Visibility, Destroy,
        Leave and DETAIL is the button number for ButtonPress,
        ButtonRelease and DETAIL is the Keysym for KeyPress and
        KeyRelease. Examples are
        <Control-Button-1> for pressing Control and mouse button 1 or
        <Alt-A> for pressing A and the Alt key (KeyPress can be omitted).
        An event pattern can also be a virtual event of the form
        <<AString>> where AString can be arbitrary. This
        event can be generated by event_generate.
        If events are concatenated they must appear shortly
        after each other.

        FUNC will be called if the event sequence occurs with an
        instance of Event as argument. If the return value of FUNC is
        "break" no further bound function is invoked.

        An additional boolean parameter ADD specifies whether FUNC will
        be called additionally to the other bound function or whether
        it will replace the previous function.

        Bind will return an identifier to allow deletion of the bound function with
        unbind without memory leak.

        If FUNC or SEQUENCE is omitted the bound function or list
        of bound events are returned."""

        return self._bind(('bind', self._w), sequence, func, add)

    def unbind(self, sequence, funcid=None):
        """Unbind for this widget the event SEQUENCE.

        If FUNCID is given, only unbind the function identified with FUNCID
        and also delete the corresponding Tcl command.

        Otherwise destroy the current binding for SEQUENCE, leaving SEQUENCE
        unbound.
        """
        self._unbind(('bind', self._w, sequence), funcid)

    def _unbind(self, what, funcid=None):
        if funcid is None:
            self.tk.call(*what, '')
        else:
            lines = self.tk.call(what).split('\n')
            prefix = f'if {{"[{funcid} '
            keep = '\n'.join(line for line in lines
                             if not line.startswith(prefix))
            if not keep.strip():
                keep = ''
            self.tk.call(*what, keep)
            self.deletecommand(funcid)

    def bind_all(self, sequence=None, func=None, add=None):
        """Bind to all widgets at an event SEQUENCE a call to function FUNC.
        An additional boolean parameter ADD specifies whether FUNC will
        be called additionally to the other bound function or whether
        it will replace the previous function. See bind for the return value."""
        return self._root()._bind(('bind', 'all'), sequence, func, add, True)

    def unbind_all(self, sequence):
        """Unbind for all widgets for event SEQUENCE all functions."""
        self._root()._unbind(('bind', 'all', sequence))

    def bind_class(self, className, sequence=None, func=None, add=None):
        """Bind to widgets with bindtag CLASSNAME at event
        SEQUENCE a call of function FUNC. An additional
        boolean parameter ADD specifies whether FUNC will be
        called additionally to the other bound function or
        whether it will replace the previous function. See bind for
        the return value."""

        return self._root()._bind(('bind', className), sequence, func, add, True)

    def unbind_class(self, className, sequence):
        """Unbind for all widgets with bindtag CLASSNAME for event SEQUENCE
        all functions."""
        self._root()._unbind(('bind', className, sequence))

    def mainloop(self, n=0):
        """Call the mainloop of Tk."""
        self.tk.mainloop(n)

    def quit(self):
        """Quit the Tcl interpreter. All widgets will be destroyed."""
        self.tk.quit()

    def _getints(self, string):
        """Internal function."""
        if string:
            return tuple(map(self.tk.getint, self.tk.splitlist(string)))

    def _getdoubles(self, string):
        """Internal function."""
        if string:
            return tuple(map(self.tk.getdouble, self.tk.splitlist(string)))

    def _getboolean(self, string):
        """Internal function."""
        if string:
            return self.tk.getboolean(string)

    def _displayof(self, displayof):
        """Internal function."""
        if displayof:
            return ('-displayof', displayof)
        if displayof is None:
            return ('-displayof', self._w)
        return ()

    @property
    def _windowingsystem(self):
        """Internal function."""
        try:
            return self._root()._windowingsystem_cached
        except AttributeError:
            ws = self._root()._windowingsystem_cached = \
                        self.tk.call('tk', 'windowingsystem')
            return ws

    def _options(self, cnf, kw = None):
        """Internal function."""
        if kw:
            cnf = _cnfmerge((cnf, kw))
        else:
            cnf = _cnfmerge(cnf)
        res = ()
        for k, v in cnf.items():
            if v is not None:
                if k[-1] == '_': k = k[:-1]
                if callable(v):
                    v = self._register(v)
                elif isinstance(v, (tuple, list)):
                    nv = []
                    for item in v:
                        if isinstance(item, int):
                            nv.append(str(item))
                        elif isinstance(item, str):
                            nv.append(_stringify(item))
                        else:
                            break
                    else:
                        v = ' '.join(nv)
                res = res + ('-'+k, v)
        return res

    def nametowidget(self, name):
        """Return the Tkinter instance of a widget identified by
        its Tcl name NAME."""
        name = str(name).split('.')
        w = self

        if not name[0]:
            w = w._root()
            name = name[1:]

        for n in name:
            if not n:
                break
            w = w.children[n]

        return w

    _nametowidget = nametowidget

    def _register(self, func, subst=None, needcleanup=1):
        """Return a newly created Tcl function. If this
        function is called, the Python function FUNC will
        be executed. An optional function SUBST can
        be given which will be executed before FUNC."""
        f = CallWrapper(func, subst, self).__call__
        name = repr(id(f))
        try:
            func = func.__func__
        except AttributeError:
            pass
        try:
            name = name + func.__name__
        except AttributeError:
            pass
        self.tk.createcommand(name, f)
        if needcleanup:
            if self._tclCommands is None:
                self._tclCommands = []
            self._tclCommands.append(name)
        return name

    register = _register

    def _root(self):
        """Internal function."""
        w = self
        while w.master is not None: w = w.master
        return w
    _subst_format = ('%#', '%b', '%f', '%h', '%k',
             '%s', '%t', '%w', '%x', '%y',
             '%A', '%E', '%K', '%N', '%W', '%T', '%X', '%Y', '%D')
    _subst_format_str = " ".join(_subst_format)

    def _substitute(self, *args):
        """Internal function."""
        if len(args) != len(self._subst_format): return args
        getboolean = self.tk.getboolean

        getint = self.tk.getint
        def getint_event(s):
            """Tk changed behavior in 8.4.2, returning "??" rather more often."""
            try:
                return getint(s)
            except (ValueError, TclError):
                return s

        nsign, b, f, h, k, s, t, w, x, y, A, E, K, N, W, T, X, Y, D = args
        # Missing: (a, c, d, m, o, v, B, R)
        e = Event()
        # serial field: valid for all events
        # number of button: ButtonPress and ButtonRelease events only
        # height field: Configure, ConfigureRequest, Create,
        # ResizeRequest, and Expose events only
        # keycode field: KeyPress and KeyRelease events only
        # time field: "valid for events that contain a time field"
        # width field: Configure, ConfigureRequest, Create, ResizeRequest,
        # and Expose events only
        # x field: "valid for events that contain an x field"
        # y field: "valid for events that contain a y field"
        # keysym as decimal: KeyPress and KeyRelease events only
        # x_root, y_root fields: ButtonPress, ButtonRelease, KeyPress,
        # KeyRelease, and Motion events
        e.serial = getint(nsign)
        e.num = getint_event(b)
        try: e.focus = getboolean(f)
        except TclError: pass
        e.height = getint_event(h)
        e.keycode = getint_event(k)
        e.state = getint_event(s)
        e.time = getint_event(t)
        e.width = getint_event(w)
        e.x = getint_event(x)
        e.y = getint_event(y)
        e.char = A
        try: e.send_event = getboolean(E)
        except TclError: pass
        e.keysym = K
        e.keysym_num = getint_event(N)
        try:
            e.type = EventType(T)
        except ValueError:
            e.type = T
        try:
            e.widget = self._nametowidget(W)
        except KeyError:
            e.widget = W
        e.x_root = getint_event(X)
        e.y_root = getint_event(Y)
        try:
            e.delta = getint(D)
        except (ValueError, TclError):
            e.delta = 0
        return (e,)

    def _report_exception(self):
        """Internal function."""
        exc, val, tb = sys.exc_info()
        root = self._root()
        root.report_callback_exception(exc, val, tb)

    def _getconfigure(self, *args):
        """Call Tcl configure command and return the result as a dict."""
        cnf = {}
        for x in self.tk.splitlist(self.tk.call(*args)):
            x = self.tk.splitlist(x)
            cnf[x[0][1:]] = (x[0][1:],) + x[1:]
        return cnf

    def _getconfigure1(self, *args):
        x = self.tk.splitlist(self.tk.call(*args))
        return (x[0][1:],) + x[1:]

    def _configure(self, cmd, cnf, kw):
        """Internal function."""
        if kw:
            cnf = _cnfmerge((cnf, kw))
        elif cnf:
            cnf = _cnfmerge(cnf)
        if cnf is None:
            return self._getconfigure(_flatten((self._w, cmd)))
        if isinstance(cnf, str):
            return self._getconfigure1(_flatten((self._w, cmd, '-'+cnf)))
        self.tk.call(_flatten((self._w, cmd)) + self._options(cnf))
    # These used to be defined in Widget:

    def configure(self, cnf=None, **kw):
        """Configure resources of a widget.

        The values for resources are specified as keyword
        arguments. To get an overview about
        the allowed keyword arguments call the method keys.
        """
        return self._configure('configure', cnf, kw)

    config = configure

    def cget(self, key):
        """Return the resource value for a KEY given as string."""
        return self.tk.call(self._w, 'cget', '-' + key)

    __getitem__ = cget

    def __setitem__(self, key, value):
        self.configure({key: value})

    def keys(self):
        """Return a list of all resource names of this widget."""
        splitlist = self.tk.splitlist
        return [splitlist(x)[0][1:] for x in
                splitlist(self.tk.call(self._w, 'configure'))]

    def __str__(self):
        """Return the window path name of this widget."""
        return self._w

    def __repr__(self):
        return '<%s.%s object %s>' % (
            self.__class__.__module__, self.__class__.__qualname__, self._w)

    # Pack methods that apply to the master
    _noarg_ = ['_noarg_']

    def pack_propagate(self, flag=_noarg_):
        """Set or get the status for propagation of geometry information.

        A boolean argument specifies whether the geometry information
        of the slaves will determine the size of this widget. If no argument
        is given the current setting will be returned.
        """
        if flag is Misc._noarg_:
            return self._getboolean(self.tk.call(
                'pack', 'propagate', self._w))
        else:
            self.tk.call('pack', 'propagate', self._w, flag)

    propagate = pack_propagate

    def pack_slaves(self):
        """Return a list of all slaves of this widget
        in its packing order."""
        return [self._nametowidget(x) for x in
                self.tk.splitlist(
                   self.tk.call('pack', 'slaves', self._w))]

    slaves = pack_slaves

    # Place method that applies to the master
    def place_slaves(self):
        """Return a list of all slaves of this widget
        in its packing order."""
        return [self._nametowidget(x) for x in
                self.tk.splitlist(
                   self.tk.call(
                       'place', 'slaves', self._w))]

    # Grid methods that apply to the master

    def grid_anchor(self, anchor=None): # new in Tk 8.5
        """The anchor value controls how to place the grid within the
        master when no row/column has any weight.

        The default anchor is nw."""
        self.tk.call('grid', 'anchor', self._w, anchor)

    anchor = grid_anchor

    def grid_bbox(self, column=None, row=None, col2=None, row2=None):
        """Return a tuple of integer coordinates for the bounding
        box of this widget controlled by the geometry manager grid.

        If COLUMN, ROW is given the bounding box applies from
        the cell with row and column 0 to the specified
        cell. If COL2 and ROW2 are given the bounding box
        starts at that cell.

        The returned integers specify the offset of the upper left
        corner in the master widget and the width and height.
        """
        args = ('grid', 'bbox', self._w)
        if column is not None and row is not None:
            args = args + (column, row)
        if col2 is not None and row2 is not None:
            args = args + (col2, row2)
        return self._getints(self.tk.call(*args)) or None

    bbox = grid_bbox

    def _gridconvvalue(self, value):
        if isinstance(value, (str, _tkinter.Tcl_Obj)):
            try:
                svalue = str(value)
                if not svalue:
                    return None
                elif '.' in svalue:
                    return self.tk.getdouble(svalue)
                else:
                    return self.tk.getint(svalue)
            except (ValueError, TclError):
                pass
        return value

    def _grid_configure(self, command, index, cnf, kw):
        """Internal function."""
        if isinstance(cnf, str) and not kw:
            if cnf[-1:] == '_':
                cnf = cnf[:-1]
            if cnf[:1] != '-':
                cnf = '-'+cnf
            options = (cnf,)
        else:
            options = self._options(cnf, kw)
        if not options:
            return _splitdict(
                self.tk,
                self.tk.call('grid', command, self._w, index),
                conv=self._gridconvvalue)
        res = self.tk.call(
                  ('grid', command, self._w, index)
                  + options)
        if len(options) == 1:
            return self._gridconvvalue(res)

    def grid_columnconfigure(self, index, cnf={}, **kw):
        """Configure column INDEX of a grid.

        Valid resources are minsize (minimum size of the column),
        weight (how much does additional space propagate to this column)
        and pad (how much space to let additionally)."""
        return self._grid_configure('columnconfigure', index, cnf, kw)

    columnconfigure = grid_columnconfigure

    def grid_location(self, x, y):
        """Return a tuple of column and row which identify the cell
        at which the pixel at position X and Y inside the master
        widget is located."""
        return self._getints(
            self.tk.call(
                'grid', 'location', self._w, x, y)) or None

    def grid_propagate(self, flag=_noarg_):
        """Set or get the status for propagation of geometry information.

        A boolean argument specifies whether the geometry information
        of the slaves will determine the size of this widget. If no argument
        is given, the current setting will be returned.
        """
        if flag is Misc._noarg_:
            return self._getboolean(self.tk.call(
                'grid', 'propagate', self._w))
        else:
            self.tk.call('grid', 'propagate', self._w, flag)

    def grid_rowconfigure(self, index, cnf={}, **kw):
        """Configure row INDEX of a grid.

        Valid resources are minsize (minimum size of the row),
        weight (how much does additional space propagate to this row)
        and pad (how much space to let additionally)."""
        return self._grid_configure('rowconfigure', index, cnf, kw)

    rowconfigure = grid_rowconfigure

    def grid_size(self):
        """Return a tuple of the number of column and rows in the grid."""
        return self._getints(
            self.tk.call('grid', 'size', self._w)) or None

    size = grid_size

    def grid_slaves(self, row=None, column=None):
        """Return a list of all slaves of this widget
        in its packing order."""
        args = ()
        if row is not None:
            args = args + ('-row', row)
        if column is not None:
            args = args + ('-column', column)
        return [self._nametowidget(x) for x in
                self.tk.splitlist(self.tk.call(
                   ('grid', 'slaves', self._w) + args))]

    # Support for the "event" command, new in Tk 4.2.
    # By Case Roole.

    def event_add(self, virtual, *sequences):
        """Bind a virtual event VIRTUAL (of the form <<Name>>)
        to an event SEQUENCE such that the virtual event is triggered
        whenever SEQUENCE occurs."""
        args = ('event', 'add', virtual) + sequences
        self.tk.call(args)

    def event_delete(self, virtual, *sequences):
        """Unbind a virtual event VIRTUAL from SEQUENCE."""
        args = ('event', 'delete', virtual) + sequences
        self.tk.call(args)

    def event_generate(self, sequence, **kw):
        """Generate an event SEQUENCE. Additional
        keyword arguments specify parameter of the event
        (e.g. x, y, rootx, rooty)."""
        args = ('event', 'generate', self._w, sequence)
        for k, v in kw.items():
            args = args + ('-%s' % k, str(v))
        self.tk.call(args)

    def event_info(self, virtual=None):
        """Return a list of all virtual events or the information
        about the SEQUENCE bound to the virtual event VIRTUAL."""
        return self.tk.splitlist(
            self.tk.call('event', 'info', virtual))

    # Image related commands

    def image_names(self):
        """Return a list of all existing image names."""
        return self.tk.splitlist(self.tk.call('image', 'names'))

    def image_types(self):
        """Return a list of all available image types (e.g. photo bitmap)."""
        return self.tk.splitlist(self.tk.call('image', 'types'))


class CallWrapper:
    """Internal class. Stores function to call when some user
    defined Tcl function is called e.g. after an event occurred."""

    def __init__(self, func, subst, widget):
        """Store FUNC, SUBST and WIDGET as members."""
        self.func = func
        self.subst = subst
        self.widget = widget

    def __call__(self, *args):
        """Apply first function SUBST to arguments, than FUNC."""
        try:
            if self.subst:
                args = self.subst(*args)
            return self.func(*args)
        except SystemExit:
            raise
        except:
            self.widget._report_exception()


class XView:
    """Mix-in class for querying and changing the horizontal position
    of a widget's window."""

    def xview(self, *args):
        """Query and change the horizontal position of the view."""
        res = self.tk.call(self._w, 'xview', *args)
        if not args:
            return self._getdoubles(res)

    def xview_moveto(self, fraction):
        """Adjusts the view in the window so that FRACTION of the
        total width of the canvas is off-screen to the left."""
        self.tk.call(self._w, 'xview', 'moveto', fraction)

    def xview_scroll(self, number, what):
        """Shift the x-view according to NUMBER which is measured in "units"
        or "pages" (WHAT)."""
        self.tk.call(self._w, 'xview', 'scroll', number, what)


class YView:
    """Mix-in class for querying and changing the vertical position
    of a widget's window."""

    def yview(self, *args):
        """Query and change the vertical position of the view."""
        res = self.tk.call(self._w, 'yview', *args)
        if not args:
            return self._getdoubles(res)

    def yview_moveto(self, fraction):
        """Adjusts the view in the window so that FRACTION of the
        total height of the canvas is off-screen to the top."""
        self.tk.call(self._w, 'yview', 'moveto', fraction)

    def yview_scroll(self, number, what):
        """Shift the y-view according to NUMBER which is measured in
        "units" or "pages" (WHAT)."""
        self.tk.call(self._w, 'yview', 'scroll', number, what)


class Wm:
    """Provides functions for the communication with the window manager."""

    def wm_aspect(self,
              minNumer=None, minDenom=None,
              maxNumer=None, maxDenom=None):
        """Instruct the window manager to set the aspect ratio (width/height)
        of this widget to be between MINNUMER/MINDENOM and MAXNUMER/MAXDENOM. Return a tuple
        of the actual values if no argument is given."""
        return self._getints(
            self.tk.call('wm', 'aspect', self._w,
                     minNumer, minDenom,
                     maxNumer, maxDenom))

    aspect = wm_aspect

    def wm_attributes(self, *args):
        """This subcommand returns or sets platform specific attributes

        The first form returns a list of the platform specific flags and
        their values. The second form returns the value for the specific
        option. The third form sets one or more of the values. The values
        are as follows:

        On Windows, -disabled gets or sets whether the window is in a
        disabled state. -toolwindow gets or sets the style of the window
        to toolwindow (as defined in the MSDN). -topmost gets or sets
        whether this is a topmost window (displays above all other
        windows).

        On Macintosh, XXXXX

        On Unix, there are currently no special attribute values.
        """
        args = ('wm', 'attributes', self._w) + args
        return self.tk.call(args)

    attributes = wm_attributes

    def wm_client(self, name=None):
        """Store NAME in WM_CLIENT_MACHINE property of this widget. Return
        current value."""
        return self.tk.call('wm', 'client', self._w, name)

    client = wm_client

    def wm_colormapwindows(self, *wlist):
        """Store list of window names (WLIST) into WM_COLORMAPWINDOWS property
        of this widget. This list contains windows whose colormaps differ from their
        parents. Return current list of widgets if WLIST is empty."""
        if len(wlist) > 1:
            wlist = (wlist,) # Tk needs a list of windows here
        args = ('wm', 'colormapwindows', self._w) + wlist
        if wlist:
            self.tk.call(args)
        else:
            return [self._nametowidget(x)
                    for x in self.tk.splitlist(self.tk.call(args))]

    colormapwindows = wm_colormapwindows

    def wm_command(self, value=None):
        """Store VALUE in WM_COMMAND property. It is the command
        which shall be used to invoke the application. Return current
        command if VALUE is None."""
        return self.tk.call('wm', 'command', self._w, value)

    command = wm_command

    def wm_deiconify(self):
        """Deiconify this widget. If it was never mapped it will not be mapped.
        On Windows it will raise this widget and give it the focus."""
        return self.tk.call('wm', 'deiconify', self._w)

    deiconify = wm_deiconify

    def wm_focusmodel(self, model=None):
        """Set focus model to MODEL. "active" means that this widget will claim
        the focus itself, "passive" means that the window manager shall give
        the focus. Return current focus model if MODEL is None."""
        return self.tk.call('wm', 'focusmodel', self._w, model)

    focusmodel = wm_focusmodel

    def wm_forget(self, window): # new in Tk 8.5
        """The window will be unmapped from the screen and will no longer
        be managed by wm. toplevel windows will be treated like frame
        windows once they are no longer managed by wm, however, the menu
        option configuration will be remembered and the menus will return
        once the widget is managed again."""
        self.tk.call('wm', 'forget', window)

    forget = wm_forget

    def wm_frame(self):
        """Return identifier for decorative frame of this widget if present."""
        return self.tk.call('wm', 'frame', self._w)

    frame = wm_frame

    def wm_geometry(self, newGeometry=None):
        """Set geometry to NEWGEOMETRY of the form =widthxheight+x+y. Return
        current value if None is given."""
        return self.tk.call('wm', 'geometry', self._w, newGeometry)

    geometry = wm_geometry

    def wm_grid(self,
         baseWidth=None, baseHeight=None,
         widthInc=None, heightInc=None):
        """Instruct the window manager that this widget shall only be
        resized on grid boundaries. WIDTHINC and HEIGHTINC are the width and
        height of a grid unit in pixels. BASEWIDTH and BASEHEIGHT are the
        number of grid units requested in Tk_GeometryRequest."""
        return self._getints(self.tk.call(
            'wm', 'grid', self._w,
            baseWidth, baseHeight, widthInc, heightInc))

    grid = wm_grid

    def wm_group(self, pathName=None):
        """Set the group leader widgets for related widgets to PATHNAME. Return
        the group leader of this widget if None is given."""
        return self.tk.call('wm', 'group', self._w, pathName)

    group = wm_group

    def wm_iconbitmap(self, bitmap=None, default=None):
        """Set bitmap for the iconified widget to BITMAP. Return
        the bitmap if None is given.

        Under Windows, the DEFAULT parameter can be used to set the icon
        for the widget and any descendants that don't have an icon set
        explicitly.  DEFAULT can be the relative path to a .ico file
        (example: root.iconbitmap(default='myicon.ico') ).  See Tk
        documentation for more information."""
        if default:
            return self.tk.call('wm', 'iconbitmap', self._w, '-default', default)
        else:
            return self.tk.call('wm', 'iconbitmap', self._w, bitmap)

    iconbitmap = wm_iconbitmap

    def wm_iconify(self):
        """Display widget as icon."""
        return self.tk.call('wm', 'iconify', self._w)

    iconify = wm_iconify

    def wm_iconmask(self, bitmap=None):
        """Set mask for the icon bitmap of this widget. Return the
        mask if None is given."""
        return self.tk.call('wm', 'iconmask', self._w, bitmap)

    iconmask = wm_iconmask

    def wm_iconname(self, newName=None):
        """Set the name of the icon for this widget. Return the name if
        None is given."""
        return self.tk.call('wm', 'iconname', self._w, newName)

    iconname = wm_iconname

    def wm_iconphoto(self, default=False, *args): # new in Tk 8.5
        """Sets the titlebar icon for this window based on the named photo
        images passed through args. If default is True, this is applied to
        all future created toplevels as well.

        The data in the images is taken as a snapshot at the time of
        invocation. If the images are later changed, this is not reflected
        to the titlebar icons. Multiple images are accepted to allow
        different images sizes to be provided. The window manager may scale
        provided icons to an appropriate size.

        On Windows, the images are packed into a Windows icon structure.
        This will override an icon specified to wm_iconbitmap, and vice
        versa.

        On X, the images are arranged into the _NET_WM_ICON X property,
        which most modern window managers support. An icon specified by
        wm_iconbitmap may exist simultaneously.

        On Macintosh, this currently does nothing."""
        if default:
            self.tk.call('wm', 'iconphoto', self._w, "-default", *args)
        else:
            self.tk.call('wm', 'iconphoto', self._w, *args)

    iconphoto = wm_iconphoto

    def wm_iconposition(self, x=None, y=None):
        """Set the position of the icon of this widget to X and Y. Return
        a tuple of the current values of X and X if None is given."""
        return self._getints(self.tk.call(
            'wm', 'iconposition', self._w, x, y))

    iconposition = wm_iconposition

    def wm_iconwindow(self, pathName=None):
        """Set widget PATHNAME to be displayed instead of icon. Return the current
        value if None is given."""
        return self.tk.call('wm', 'iconwindow', self._w, pathName)

    iconwindow = wm_iconwindow

    def wm_manage(self, widget): # new in Tk 8.5
        """The widget specified will become a stand alone top-level window.
        The window will be decorated with the window managers title bar,
        etc."""
        self.tk.call('wm', 'manage', widget)

    manage = wm_manage

    def wm_maxsize(self, width=None, height=None):
        """Set max WIDTH and HEIGHT for this widget. If the window is gridded
        the values are given in grid units. Return the current values if None
        is given."""
        return self._getints(self.tk.call(
            'wm', 'maxsize', self._w, width, height))

    maxsize = wm_maxsize

    def wm_minsize(self, width=None, height=None):
        """Set min WIDTH and HEIGHT for this widget. If the window is gridded
        the values are given in grid units. Return the current values if None
        is given."""
        return self._getints(self.tk.call(
            'wm', 'minsize', self._w, width, height))

    minsize = wm_minsize

    def wm_overrideredirect(self, boolean=None):
        """Instruct the window manager to ignore this widget
        if BOOLEAN is given with 1. Return the current value if None
        is given."""
        return self._getboolean(self.tk.call(
            'wm', 'overrideredirect', self._w, boolean))

    overrideredirect = wm_overrideredirect

    def wm_positionfrom(self, who=None):
        """Instruct the window manager that the position of this widget shall
        be defined by the user if WHO is "user", and by its own policy if WHO is
        "program"."""
        return self.tk.call('wm', 'positionfrom', self._w, who)

    positionfrom = wm_positionfrom

    def wm_protocol(self, name=None, func=None):
        """Bind function FUNC to command NAME for this widget.
        Return the function bound to NAME if None is given. NAME could be
        e.g. "WM_SAVE_YOURSELF" or "WM_DELETE_WINDOW"."""
        if callable(func):
            command = self._register(func)
        else:
            command = func
        return self.tk.call(
            'wm', 'protocol', self._w, name, command)

    protocol = wm_protocol

    def wm_resizable(self, width=None, height=None):
        """Instruct the window manager whether this width can be resized
        in WIDTH or HEIGHT. Both values are boolean values."""
        return self.tk.call('wm', 'resizable', self._w, width, height)

    resizable = wm_resizable

    def wm_sizefrom(self, who=None):
        """Instruct the window manager that the size of this widget shall
        be defined by the user if WHO is "user", and by its own policy if WHO is
        "program"."""
        return self.tk.call('wm', 'sizefrom', self._w, who)

    sizefrom = wm_sizefrom

    def wm_state(self, newstate=None):
        """Query or set the state of this widget as one of normal, icon,
        iconic (see wm_iconwindow), withdrawn, or zoomed (Windows only)."""
        return self.tk.call('wm', 'state', self._w, newstate)

    state = wm_state

    def wm_title(self, string=None):
        """Set the title of this widget."""
        return self.tk.call('wm', 'title', self._w, string)

    title = wm_title

    def wm_transient(self, master=None):
        """Instruct the window manager that this widget is transient
        with regard to widget MASTER."""
        return self.tk.call('wm', 'transient', self._w, master)

    transient = wm_transient

    def wm_withdraw(self):
        """Withdraw this widget from the screen such that it is unmapped
        and forgotten by the window manager. Re-draw it with wm_deiconify."""
        return self.tk.call('wm', 'withdraw', self._w)

    withdraw = wm_withdraw


class Tk(Misc, Wm):
    """Toplevel widget of Tk which represents mostly the main window
    of an application. It has an associated Tcl interpreter."""
    _w = '.'

    def __init__(self, screenName=None, baseName=None, className='Tk',
                 useTk=True, sync=False, use=None):
        """Return a new top level widget on screen SCREENNAME. A new Tcl interpreter will
        be created. BASENAME will be used for the identification of the profile file (see
        readprofile).
        It is constructed from sys.argv[0] without extensions if None is given. CLASSNAME
        is the name of the widget class."""
        self.master = None
        self.children = {}
        self._tkloaded = False
        # to avoid recursions in the getattr code in case of failure, we
        # ensure that self.tk is always _something_.
        self.tk = None
        if baseName is None:
            import os
            baseName = os.path.basename(sys.argv[0])
            baseName, ext = os.path.splitext(baseName)
            if ext not in ('.py', '.pyc'):
                baseName = baseName + ext
        interactive = False
        self.tk = _tkinter.create(screenName, baseName, className, interactive, wantobjects, useTk, sync, use)
        if useTk:
            self._loadtk()
        if not sys.flags.ignore_environment:
            # Issue #16248: Honor the -E flag to avoid code injection.
            self.readprofile(baseName, className)

    def loadtk(self):
        if not self._tkloaded:
            self.tk.loadtk()
            self._loadtk()

    def _loadtk(self):
        self._tkloaded = True
        global _default_root
        # Version sanity checks
        tk_version = self.tk.getvar('tk_version')
        if tk_version != _tkinter.TK_VERSION:
            raise RuntimeError("tk.h version (%s) doesn't match libtk.a version (%s)"
                               % (_tkinter.TK_VERSION, tk_version))
        # Under unknown circumstances, tcl_version gets coerced to float
        tcl_version = str(self.tk.getvar('tcl_version'))
        if tcl_version != _tkinter.TCL_VERSION:
            raise RuntimeError("tcl.h version (%s) doesn't match libtcl.a version (%s)" \
                               % (_tkinter.TCL_VERSION, tcl_version))
        # Create and register the tkerror and exit commands
        # We need to inline parts of _register here, _ register
        # would register differently-named commands.
        if self._tclCommands is None:
            self._tclCommands = []
        self.tk.createcommand('tkerror', _tkerror)
        self.tk.createcommand('exit', _exit)
        self._tclCommands.append('tkerror')
        self._tclCommands.append('exit')
        if _support_default_root and _default_root is None:
            _default_root = self
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def destroy(self):
        """Destroy this and all descendants widgets. This will
        end the application of this Tcl interpreter."""
        for c in list(self.children.values()): c.destroy()
        self.tk.call('destroy', self._w)
        Misc.destroy(self)
        global _default_root
        if _support_default_root and _default_root is self:
            _default_root = None

    def readprofile(self, baseName, className):
        """Internal function. It reads .BASENAME.tcl and .CLASSNAME.tcl into
        the Tcl Interpreter and calls exec on the contents of .BASENAME.py and
        .CLASSNAME.py if such a file exists in the home directory."""
        import os
        if 'HOME' in os.environ: home = os.environ['HOME']
        else: home = os.curdir
        class_tcl = os.path.join(home, '.%s.tcl' % className)
        class_py = os.path.join(home, '.%s.py' % className)
        base_tcl = os.path.join(home, '.%s.tcl' % baseName)
        base_py = os.path.join(home, '.%s.py' % baseName)
        dir = {'self': self}
        exec('from tkinter import *', dir)
        if os.path.isfile(class_tcl):
            self.tk.call('source', class_tcl)
        if os.path.isfile(class_py):
            exec(open(class_py).read(), dir)
        if os.path.isfile(base_tcl):
            self.tk.call('source', base_tcl)
        if os.path.isfile(base_py):
            exec(open(base_py).read(), dir)

    def report_callback_exception(self, exc, val, tb):
        """Report callback exception on sys.stderr.

        Applications may want to override this internal function, and
        should when sys.stderr is None."""
        import traceback
        print("Exception in Tkinter callback", file=sys.stderr)
        sys.last_type = exc
        sys.last_value = val
        sys.last_traceback = tb
        traceback.print_exception(exc, val, tb)

    def __getattr__(self, attr):
        "Delegate attribute access to the interpreter object"
        return getattr(self.tk, attr)

# Ideally, the classes Pack, Place and Grid disappear, the
# pack/place/grid methods are defined on the Widget class, and
# everybody uses w.pack_whatever(...) instead of Pack.whatever(w,
# ...), with pack(), place() and grid() being short for
# pack_configure(), place_configure() and grid_columnconfigure(), and
# forget() being short for pack_forget().  As a practical matter, I'm
# afraid that there is too much code out there that may be using the
# Pack, Place or Grid class, so I leave them intact -- but only as
# backwards compatibility features.  Also note that those methods that
# take a master as argument (e.g. pack_propagate) have been moved to
# the Misc class (which now incorporates all methods common between
# toplevel and interior widgets).  Again, for compatibility, these are
# copied into the Pack, Place or Grid class.


def Tcl(screenName=None, baseName=None, className='Tk', useTk=False):
    return Tk(screenName, baseName, className, useTk)


class Pack:
    """Geometry manager Pack.

    Base class to use the methods pack_* in every widget."""

    def pack_configure(self, cnf={}, **kw):
        """Pack a widget in the parent widget. Use as options:
        after=widget - pack it after you have packed widget
        anchor=NSEW (or subset) - position widget according to
                                  given direction
        before=widget - pack it before you will pack widget
        expand=bool - expand widget if parent size grows
        fill=NONE or X or Y or BOTH - fill widget if widget grows
        in=master - use master to contain this widget
        in_=master - see 'in' option description
        ipadx=amount - add internal padding in x direction
        ipady=amount - add internal padding in y direction
        padx=amount - add padding in x direction
        pady=amount - add padding in y direction
        side=TOP or BOTTOM or LEFT or RIGHT -  where to add this widget.
        """
        self.tk.call(
              ('pack', 'configure', self._w)
              + self._options(cnf, kw))

    pack = configure = config = pack_configure

    def pack_forget(self):
        """Unmap this widget and do not use it for the packing order."""
        self.tk.call('pack', 'forget', self._w)

    forget = pack_forget

    def pack_info(self):
        """Return information about the packing options
        for this widget."""
        d = _splitdict(self.tk, self.tk.call('pack', 'info', self._w))
        if 'in' in d:
            d['in'] = self.nametowidget(d['in'])
        return d

    info = pack_info
    propagate = pack_propagate = Misc.pack_propagate
    slaves = pack_slaves = Misc.pack_slaves


class Place:
    """Geometry manager Place.

    Base class to use the methods place_* in every widget."""

    def place_configure(self, cnf={}, **kw):
        """Place a widget in the parent widget. Use as options:
        in=master - master relative to which the widget is placed
        in_=master - see 'in' option description
        x=amount - locate anchor of this widget at position x of master
        y=amount - locate anchor of this widget at position y of master
        relx=amount - locate anchor of this widget between 0.0 and 1.0
                      relative to width of master (1.0 is right edge)
        rely=amount - locate anchor of this widget between 0.0 and 1.0
                      relative to height of master (1.0 is bottom edge)
        anchor=NSEW (or subset) - position anchor according to given direction
        width=amount - width of this widget in pixel
        height=amount - height of this widget in pixel
        relwidth=amount - width of this widget between 0.0 and 1.0
                          relative to width of master (1.0 is the same width
                          as the master)
        relheight=amount - height of this widget between 0.0 and 1.0
                           relative to height of master (1.0 is the same
                           height as the master)
        bordermode="inside" or "outside" - whether to take border width of
                                           master widget into account
        """
        self.tk.call(
              ('place', 'configure', self._w)
              + self._options(cnf, kw))

    place = configure = config = place_configure

    def place_forget(self):
        """Unmap this widget."""
        self.tk.call('place', 'forget', self._w)

    forget = place_forget

    def place_info(self):
        """Return information about the placing options
        for this widget."""
        d = _splitdict(self.tk, self.tk.call('place', 'info', self._w))
        if 'in' in d:
            d['in'] = self.nametowidget(d['in'])
        return d

    info = place_info
    slaves = place_slaves = Misc.place_slaves


class Grid:
    """Geometry manager Grid.

    Base class to use the methods grid_* in every widget."""
    # Thanks to Masazumi Yoshikawa (yosikawa@isi.edu)

    def grid_configure(self, cnf={}, **kw):
        """Position a widget in the parent widget in a grid. Use as options:
        column=number - use cell identified with given column (starting with 0)
        columnspan=number - this widget will span several columns
        in=master - use master to contain this widget
        in_=master - see 'in' option description
        ipadx=amount - add internal padding in x direction
        ipady=amount - add internal padding in y direction
        padx=amount - add padding in x direction
        pady=amount - add padding in y direction
        row=number - use cell identified with given row (starting with 0)
        rowspan=number - this widget will span several rows
        sticky=NSEW - if cell is larger on which sides will this
                      widget stick to the cell boundary
        """
        self.tk.call(
              ('grid', 'configure', self._w)
              + self._options(cnf, kw))

    grid = configure = config = grid_configure
    bbox = grid_bbox = Misc.grid_bbox
    columnconfigure = grid_columnconfigure = Misc.grid_columnconfigure

    def grid_forget(self):
        """Unmap this widget."""
        self.tk.call('grid', 'forget', self._w)

    forget = grid_forget

    def grid_remove(self):
        """Unmap this widget but remember the grid options."""
        self.tk.call('grid', 'remove', self._w)

    def grid_info(self):
        """Return information about the options
        for positioning this widget in a grid."""
        d = _splitdict(self.tk, self.tk.call('grid', 'info', self._w))
        if 'in' in d:
            d['in'] = self.nametowidget(d['in'])
        return d

    info = grid_info
    location = grid_location = Misc.grid_location
    propagate = grid_propagate = Misc.grid_propagate
    rowconfigure = grid_rowconfigure = Misc.grid_rowconfigure
    size = grid_size = Misc.grid_size
    slaves = grid_slaves = Misc.grid_slaves


class BaseWidget(Misc):
    """Internal class."""

    def _setup(self, master, cnf):
        """Internal function. Sets up information about children."""
        if master is None:
            master = _get_default_root()
        self.master = master
        self.tk = master.tk
        name = None
        if 'name' in cnf:
            name = cnf['name']
            del cnf['name']
        if not name:
            name = self.__class__.__name__.lower()
            if master._last_child_ids is None:
                master._last_child_ids = {}
            count = master._last_child_ids.get(name, 0) + 1
            master._last_child_ids[name] = count
            if count == 1:
                name = '!%s' % (name,)
            else:
                name = '!%s%d' % (name, count)
        self._name = name
        if master._w=='.':
            self._w = '.' + name
        else:
            self._w = master._w + '.' + name
        self.children = {}
        if self._name in self.master.children:
            self.master.children[self._name].destroy()
        self.master.children[self._name] = self

    def __init__(self, master, widgetName, cnf={}, kw={}, extra=()):
        """Construct a widget with the parent widget MASTER, a name WIDGETNAME
        and appropriate options."""
        if kw:
            cnf = _cnfmerge((cnf, kw))
        self.widgetName = widgetName
        self._setup(master, cnf)
        if self._tclCommands is None:
            self._tclCommands = []
        classes = [(k, v) for k, v in cnf.items() if isinstance(k, type)]
        for k, v in classes:
            del cnf[k]
        self.tk.call(
            (widgetName, self._w) + extra + self._options(cnf))
        for k, v in classes:
            k.configure(self, v)

    def destroy(self):
        """Destroy this and all descendants widgets."""
        for c in list(self.children.values()): c.destroy()
        self.tk.call('destroy', self._w)
        if self._name in self.master.children:
            del self.master.children[self._name]
        Misc.destroy(self)

    def _do(self, name, args=()):
        # XXX Obsolete -- better use self.tk.call directly!
        return self.tk.call((self._w, name) + args)


class Widget(BaseWidget, Pack, Place, Grid):
    """Internal class.

    Base class for a widget which can be positioned with the geometry managers
    Pack, Place or Grid."""
    pass


class Toplevel(BaseWidget, Wm):
    """Toplevel widget, e.g. for dialogs."""

    def __init__(self, master=None, cnf={}, **kw):
        """Construct a toplevel widget with the parent MASTER.

        Valid resource names: background, bd, bg, borderwidth, class,
        colormap, container, cursor, height, highlightbackground,
        highlightcolor, highlightthickness, menu, relief, screen, takefocus,
        use, visual, width."""
        if kw:
            cnf = _cnfmerge((cnf, kw))
        extra = ()
        for wmkey in ['screen', 'class_', 'class', 'visual',
                  'colormap']:
            if wmkey in cnf:
                val = cnf[wmkey]
                # TBD: a hack needed because some keys
                # are not valid as keyword arguments
                if wmkey[-1] == '_': opt = '-'+wmkey[:-1]
                else: opt = '-'+wmkey
                extra = extra + (opt, val)
                del cnf[wmkey]
        BaseWidget.__init__(self, master, 'toplevel', cnf, {}, extra)
        root = self._root()
        self.iconname(root.iconname())
        self.title(root.title())
        self.protocol("WM_DELETE_WINDOW", self.destroy)


class Button(Widget):
    """Button widget."""

    def __init__(self, master=None, cnf={}, **kw):
        """Construct a button widget with the parent MASTER.

        STANDARD OPTIONS

            activebackground, activeforeground, anchor,
            background, bitmap, borderwidth, cursor,
            disabledforeground, font, foreground
            highlightbackground, highlightcolor,
            highlightthickness, image, justify,
            padx, pady, relief, repeatdelay,
            repeatinterval, takefocus, text,
            textvariable, underline, wraplength

        WIDGET-SPECIFIC OPTIONS

            command, compound, default, height,
            overrelief, state, width
        """
        Widget.__init__(self, master, 'button', cnf, kw)

    def flash(self):
        """Flash the button.

        This is accomplished by redisplaying
        the button several times, alternating between active and
        normal colors. At the end of the flash the button is left
        in the same normal/active state as when the command was
        invoked. This command is ignored if the button's state is
        disabled.
        """
        self.tk.call(self._w, 'flash')

    def invoke(self):
        """Invoke the command associated with the button.

        The return value is the return value from the command,
        or an empty string if there is no command associated with
        the button. This command is ignored if the button's state
        is disabled.
        """
        return self.tk.call(self._w, 'invoke')


class Canvas(Widget, XView, YView):
    """Canvas widget to display graphical elements like lines or text."""

    def __init__(self, master=None, cnf={}, **kw):
        """Construct a canvas widget with the parent MASTER.

        Valid resource names: background, bd, bg, borderwidth, closeenough,
        confine, cursor, height, highlightbackground, highlightcolor,
        highlightthickness, insertbackground, insertborderwidth,
        insertofftime, insertontime, insertwidth, offset, relief,
        scrollregion, selectbackground, selectborderwidth, selectforeground,
        state, takefocus, width, xscrollcommand, xscrollincrement,
        yscrollcommand, yscrollincrement."""
        Widget.__init__(self, master, 'canvas', cnf, kw)

    def addtag(self, *args):
        """Internal function."""
        self.tk.call((self._w, 'addtag') + args)

    def addtag_above(self, newtag, tagOrId):
        """Add tag NEWTAG to all items above TAGORID."""
        self.addtag(newtag, 'above', tagOrId)

    def addtag_all(self, newtag):
        """Add tag NEWTAG to all items."""
        self.addtag(newtag, 'all')

    def addtag_below(self, newtag, tagOrId):
        """Add tag NEWTAG to all items below TAGORID."""
        self.addtag(newtag, 'below', tagOrId)

    def addtag_closest(self, newtag, x, y, halo=None, start=None):
        """Add tag NEWTAG to item which is closest to pixel at X, Y.
        If several match take the top-most.
        All items closer than HALO are considered overlapping (all are
        closest). If START is specified the next below this tag is taken."""
        self.addtag(newtag, 'closest', x, y, halo, start)

    def addtag_enclosed(self, newtag, x1, y1, x2, y2):
        """Add tag NEWTAG to all items in the rectangle defined
        by X1,Y1,X2,Y2."""
        self.addtag(newtag, 'enclosed', x1, y1, x2, y2)

    def addtag_overlapping(self, newtag, x1, y1, x2, y2):
        """Add tag NEWTAG to all items which overlap the rectangle
        defined by X1,Y1,X2,Y2."""
        self.addtag(newtag, 'overlapping', x1, y1, x2, y2)

    def addtag_withtag(self, newtag, tagOrId):
        """Add tag NEWTAG to all items with TAGORID."""
        self.addtag(newtag, 'withtag', tagOrId)

    def bbox(self, *args):
        """Return a tuple of X1,Y1,X2,Y2 coordinates for a rectangle
        which encloses all items with tags specified as arguments."""
        return self._getints(
            self.tk.call((self._w, 'bbox') + args)) or None

    def tag_unbind(self, tagOrId, sequence, funcid=None):
        """Unbind for all items with TAGORID for event SEQUENCE  the
        function identified with FUNCID."""
        self._unbind((self._w, 'bind', tagOrId, sequence), funcid)

    def tag_bind(self, tagOrId, sequence=None, func=None, add=None):
        """Bind to all items with TAGORID at event SEQUENCE a call to function FUNC.

        An additional boolean parameter ADD specifies whether FUNC will be
        called additionally to the other bound function or whether it will
        replace the previous function. See bind for the return value."""
        return self._bind((self._w, 'bind', tagOrId),
                  sequence, func, add)

    def canvasx(self, screenx, gridspacing=None):
        """Return the canvas x coordinate of pixel position SCREENX rounded
        to nearest multiple of GRIDSPACING units."""
        return self.tk.getdouble(self.tk.call(
            self._w, 'canvasx', screenx, gridspacing))

    def canvasy(self, screeny, gridspacing=None):
        """Return the canvas y coordinate of pixel position SCREENY rounded
        to nearest multiple of GRIDSPACING units."""
        return self.tk.getdouble(self.tk.call(
            self._w, 'canvasy', screeny, gridspacing))

    def coords(self, *args):
        """Return a list of coordinates for the item given in ARGS."""
        # XXX Should use _flatten on args
        return [self.tk.getdouble(x) for x in
                           self.tk.splitlist(
                   self.tk.call((self._w, 'coords') + args))]

    def _create(self, itemType, args, kw): # Args: (val, val, ..., cnf={})
        """Internal function."""
        args = _flatten(args)
        cnf = args[-1]
        if isinstance(cnf, (dict, tuple)):
            args = args[:-1]
        else:
            cnf = {}
        return self.tk.getint(self.tk.call(
            self._w, 'create', itemType,
            *(args + self._options(cnf, kw))))

    def create_arc(self, *args, **kw):
        """Create arc shaped region with coordinates x1,y1,x2,y2."""
        return self._create('arc', args, kw)

    def create_bitmap(self, *args, **kw):
        """Create bitmap with coordinates x1,y1."""
        return self._create('bitmap', args, kw)

    def create_image(self, *args, **kw):
        """Create image item with coordinates x1,y1."""
        return self._create('image', args, kw)

    def create_line(self, *args, **kw):
        """Create line with coordinates x1,y1,...,xn,yn."""
        return self._create('line', args, kw)

    def create_oval(self, *args, **kw):
        """Create oval with coordinates x1,y1,x2,y2."""
        return self._create('oval', args, kw)

    def create_polygon(self, *args, **kw):
        """Create polygon with coordinates x1,y1,...,xn,yn."""
        return self._create('polygon', args, kw)

    def create_rectangle(self, *args, **kw):
        """Create rectangle with coordinates x1,y1,x2,y2."""
        return self._create('rectangle', args, kw)

    def create_text(self, *args, **kw):
        """Create text with coordinates x1,y1."""
        return self._create('text', args, kw)

    def create_window(self, *args, **kw):
        """Create window with coordinates x1,y1,x2,y2."""
        return self._create('window', args, kw)

    def dchars(self, *args):
        """Delete characters of text items identified by tag or id in ARGS (possibly
        several times) from FIRST to LAST character (including)."""
        self.tk.call((self._w, 'dchars') + args)

    def delete(self, *args):
        """Delete items identified by all tag or ids contained in ARGS."""
        self.tk.call((self._w, 'delete') + args)

    def dtag(self, *args):
        """Delete tag or id given as last arguments in ARGS from items
        identified by first argument in ARGS."""
        self.tk.call((self._w, 'dtag') + args)

    def find(self, *args):
        """Internal function."""
        return self._getints(
            self.tk.call((self._w, 'find') + args)) or ()

    def find_above(self, tagOrId):
        """Return items above TAGORID."""
        return self.find('above', tagOrId)

    def find_all(self):
        """Return all items."""
        return self.find('all')

    def find_below(self, tagOrId):
        """Return all items below TAGORID."""
        return self.find('below', tagOrId)

    def find_closest(self, x, y, halo=None, start=None):
        """Return item which is closest to pixel at X, Y.
        If several match take the top-most.
        All items closer than HALO are considered overlapping (all are
        closest). If START is specified the next below this tag is taken."""
        return self.find('closest', x, y, halo, start)

    def find_enclosed(self, x1, y1, x2, y2):
        """Return all items in rectangle defined
        by X1,Y1,X2,Y2."""
        return self.find('enclosed', x1, y1, x2, y2)

    def find_overlapping(self, x1, y1, x2, y2):
        """Return all items which overlap the rectangle
        defined by X1,Y1,X2,Y2."""
        return self.find('overlapping', x1, y1, x2, y2)

    def find_withtag(self, tagOrId):
        """Return all items with TAGORID."""
        return self.find('withtag', tagOrId)

    def focus(self, *args):
        """Set focus to the first item specified in ARGS."""
        return self.tk.call((self._w, 'focus') + args)

    def gettags(self, *args):
        """Return tags associated with the first item specified in ARGS."""
        return self.tk.splitlist(
            self.tk.call((self._w, 'gettags') + args))

    def icursor(self, *args):
        """Set cursor at position POS in the item identified by TAGORID.
        In ARGS TAGORID must be first."""
        self.tk.call((self._w, 'icursor') + args)

    def index(self, *args):
        """Return position of cursor as integer in item specified in ARGS."""
        return self.tk.getint(self.tk.call((self._w, 'index') + args))

    def insert(self, *args):
        """Insert TEXT in item TAGORID at position POS. ARGS must
        be TAGORID POS TEXT."""
        self.tk.call((self._w, 'insert') + args)

    def itemcget(self, tagOrId, option):
        """Return the resource value for an OPTION for item TAGORID."""
        return self.tk.call(
            (self._w, 'itemcget') + (tagOrId, '-'+option))

    def itemconfigure(self, tagOrId, cnf=None, **kw):
        """Configure resources of an item TAGORID.

        The values for resources are specified as keyword
        arguments. To get an overview about
        the allowed keyword arguments call the method without arguments.
        """
        return self._configure(('itemconfigure', tagOrId), cnf, kw)

    itemconfig = itemconfigure

    # lower, tkraise/lift hide Misc.lower, Misc.tkraise/lift,
    # so the preferred name for them is tag_lower, tag_raise
    # (similar to tag_bind, and similar to the Text widget);
    # unfortunately can't delete the old ones yet (maybe in 1.6)
    def tag_lower(self, *args):
        """Lower an item TAGORID given in ARGS
        (optional below another item)."""
        self.tk.call((self._w, 'lower') + args)

    lower = tag_lower

    def move(self, *args):
        """Move an item TAGORID given in ARGS."""
        self.tk.call((self._w, 'move') + args)

    def moveto(self, tagOrId, x='', y=''):
        """Move the items given by TAGORID in the canvas coordinate
        space so that the first coordinate pair of the bottommost
        item with tag TAGORID is located at position (X,Y).
        X and Y may be the empty string, in which case the
        corresponding coordinate will be unchanged. All items matching
        TAGORID remain in the same positions relative to each other."""
        self.tk.call(self._w, 'moveto', tagOrId, x, y)

    def postscript(self, cnf={}, **kw):
        """Print the contents of the canvas to a postscript
        file. Valid options: colormap, colormode, file, fontmap,
        height, pageanchor, pageheight, pagewidth, pagex, pagey,
        rotate, width, x, y."""
        return self.tk.call((self._w, 'postscript') +
                    self._options(cnf, kw))

    def tag_raise(self, *args):
        """Raise an item TAGORID given in ARGS
        (optional above another item)."""
        self.tk.call((self._w, 'raise') + args)

    lift = tkraise = tag_raise

    def scale(self, *args):
        """Scale item TAGORID with XORIGIN, YORIGIN, XSCALE, YSCALE."""
        self.tk.call((self._w, 'scale') + args)

    def scan_mark(self, x, y):
        """Remember the current X, Y coordinates."""
        self.tk.call(self._w, 'scan', 'mark', x, y)

    def scan_dragto(self, x, y, gain=10):
        """Adjust the view of the canvas to GAIN times the
        difference between X and Y and the coordinates given in
        scan_mark."""
        self.tk.call(self._w, 'scan', 'dragto', x, y, gain)

    def select_adjust(self, tagOrId, index):
        """Adjust the end of the selection near the cursor of an item TAGORID to index."""
        self.tk.call(self._w, 'select', 'adjust', tagOrId, index)

    def select_clear(self):
        """Clear the selection if it is in this widget."""
        self.tk.call(self._w, 'select', 'clear')

    def select_from(self, tagOrId, index):
        """Set the fixed end of a selection in item TAGORID to INDEX."""
        self.tk.call(self._w, 'select', 'from', tagOrId, index)

    def select_item(self):
        """Return the item which has the selection."""
        return self.tk.call(self._w, 'select', 'item') or None

    def select_to(self, tagOrId, index):
        """Set the variable end of a selection in item TAGORID to INDEX."""
        self.tk.call(self._w, 'select', 'to', tagOrId, index)

    def type(self, tagOrId):
        """Return the type of the item TAGORID."""
        return self.tk.call(self._w, 'type', tagOrId) or None


_checkbutton_count = 0

class Checkbutton(Widget):
    """Checkbutton widget which is either in on- or off-state."""

    def __init__(self, master=None, cnf={}, **kw):
        """Construct a checkbutton widget with the parent MASTER.

        Valid resource names: activebackground, activeforeground, anchor,
        background, bd, bg, bitmap, borderwidth, command, cursor,
        disabledforeground, fg, font, foreground, height,
        highlightbackground, highlightcolor, highlightthickness, image,
        indicatoron, justify, offvalue, onvalue, padx, pady, relief,
        selectcolor, selectimage, state, takefocus, text, textvariable,
        underline, variable, width, wraplength."""
        Widget.__init__(self, master, 'checkbutton', cnf, kw)

    def _setup(self, master, cnf):
        # Because Checkbutton defaults to a variable with the same name as
        # the widget, Checkbutton default names must be globally unique,
        # not just unique within the parent widget.
        if not cnf.get('name'):
            global _checkbutton_count
            name = self.__class__.__name__.lower()
            _checkbutton_count += 1
            # To avoid collisions with ttk.Checkbutton, use the different
            # name template.
            cnf['name'] = f'!{name}-{_checkbutton_count}'
        super()._setup(master, cnf)

    def deselect(self):
        """Put the button in off-state."""
        self.tk.call(self._w, 'deselect')

    def flash(self):
        """Flash the button."""
        self.tk.call(self._w, 'flash')

    def invoke(self):
        """Toggle the button and invoke a command if given as resource."""
        return self.tk.call(self._w, 'invoke')

    def select(self):
        """Put the button in on-state."""
        self.tk.call(self._w, 'select')

    def toggle(self):
        """Toggle the button."""
        self.tk.call(self._w, 'toggle')


class Entry(Widget, XView):
    """Entry widget which allows displaying simple text."""

    def __init__(self, master=None, cnf={}, **kw):
        """Construct an entry widget with the parent MASTER.

        Valid resource names: background, bd, bg, borderwidth, cursor,
        exportselection, fg, font, foreground, highlightbackground,
        highlightcolor, highlightthickness, insertbackground,
        insertborderwidth, insertofftime, insertontime, insertwidth,
        invalidcommand, invcmd, justify, relief, selectbackground,
        selectborderwidth, selectforeground, show, state, takefocus,
        textvariable, validate, validatecommand, vcmd, width,
        xscrollcommand."""
        Widget.__init__(self, master, 'entry', cnf, kw)

    def delete(self, first, last=None):
        """Delete text from FIRST to LAST (not included)."""
        self.tk.call(self._w, 'delete', first, last)

    def get(self):
        """Return the text."""
        return self.tk.call(self._w, 'get')

    def icursor(self, index):
        """Insert cursor at INDEX."""
        self.tk.call(self._w, 'icursor', index)

    def index(self, index):
        """Return position of cursor."""
        return self.tk.getint(self.tk.call(
            self._w, 'index', index))

    def insert(self, index, string):
        """Insert STRING at INDEX."""
        self.tk.call(self._w, 'insert', index, string)

    def scan_mark(self, x):
        """Remember the current X, Y coordinates."""
        self.tk.call(self._w, 'scan', 'mark', x)

    def scan_dragto(self, x):
        """Adjust the view of the canvas to 10 times the
        difference between X and Y and the coordinates given in
        scan_mark."""
        self.tk.call(self._w, 'scan', 'dragto', x)

    def selection_adjust(self, index):
        """Adjust the end of the selection near the cursor to INDEX."""
        self.tk.call(self._w, 'selection', 'adjust', index)

    select_adjust = selection_adjust

    def selection_clear(self):
        """Clear the selection if it is in this widget."""
        self.tk.call(self._w, 'selection', 'clear')

    select_clear = selection_clear

    def selection_from(self, index):
        """Set the fixed end of a selection to INDEX."""
        self.tk.call(self._w, 'selection', 'from', index)

    select_from = selection_from

    def selection_present(self):
        """Return True if there are characters selected in the entry, False
        otherwise."""
        return self.tk.getboolean(
            self.tk.call(self._w, 'selection', 'present'))

    select_present = selection_present

    def selection_range(self, start, end):
        """Set the selection from START to END (not included)."""
        self.tk.call(self._w, 'selection', 'range', start, end)

    select_range = selection_range

    def selection_to(self, index):
        """Set the variable end of a selection to INDEX."""
        self.tk.call(self._w, 'selection', 'to', index)

    select_to = selection_to


class Frame(Widget):
    """Frame widget which may contain other widgets and can have a 3D border."""

    def __init__(self, master=None, cnf={}, **kw):
        """Construct a frame widget with the parent MASTER.

        Valid resource names: background, bd, bg, borderwidth, class,
        colormap, container, cursor, height, highlightbackground,
        highlightcolor, highlightthickness, relief, takefocus, visual, width."""
        cnf = _cnfmerge((cnf, kw))
        extra = ()
        if 'class_' in cnf:
            extra = ('-class', cnf['class_'])
            del cnf['class_']
        elif 'class' in cnf:
            extra = ('-class', cnf['class'])
            del cnf['class']
        Widget.__init__(self, master, 'frame', cnf, {}, extra)


class Label(Widget):
    """Label widget which can display text and bitmaps."""

    def __init__(self, master=None, cnf={}, **kw):
        """Construct a label widget with the parent MASTER.

        STANDARD OPTIONS

            activebackground, activeforeground, anchor,
            background, bitmap, borderwidth, cursor,
            disabledforeground, font, foreground,
            highlightbackground, highlightcolor,
            highlightthickness, image, justify,
            padx, pady, relief, takefocus, text,
            textvariable, underline, wraplength

        WIDGET-SPECIFIC OPTIONS

            height, state, width

        """
        Widget.__init__(self, master, 'label', cnf, kw)


class Listbox(Widget, XView, YView):
    """Listbox widget which can display a list of strings."""

    def __init__(self, master=None, cnf={}, **kw):
        """Construct a listbox widget with the parent MASTER.

        Valid resource names: background, bd, bg, borderwidth, cursor,
        exportselection, fg, font, foreground, height, highlightbackground,
        highlightcolor, highlightthickness, relief, selectbackground,
        selectborderwidth, selectforeground, selectmode, setgrid, takefocus,
        width, xscrollcommand, yscrollcommand, listvariable."""
        Widget.__init__(self, master, 'listbox', cnf, kw)

    def activate(self, index):
        """Activate item identified by INDEX."""
        self.tk.call(self._w, 'activate', index)

    def bbox(self, index):
        """Return a tuple of X1,Y1,X2,Y2 coordinates for a rectangle
        which encloses the item identified by the given index."""
        return self._getints(self.tk.call(self._w, 'bbox', index)) or None

    def curselection(self):
        """Return the indices of currently selected item."""
        return self._getints(self.tk.call(self._w, 'curselection')) or ()

    def delete(self, first, last=None):
        """Delete items from FIRST to LAST (included)."""
        self.tk.call(self._w, 'delete', first, last)

    def get(self, first, last=None):
        """Get list of items from FIRST to LAST (included)."""
        if last is not None:
            return self.tk.splitlist(self.tk.call(
                self._w, 'get', first, last))
        else:
            return self.tk.call(self._w, 'get', first)

    def index(self, index):
        """Return index of item identified with INDEX."""
        i = self.tk.call(self._w, 'index', index)
        if i == 'none': return None
        return self.tk.getint(i)

    def insert(self, index, *elements):
        """Insert ELEMENTS at INDEX."""
        self.tk.call((self._w, 'insert', index) + elements)

    def nearest(self, y):
        """Get index of item which is nearest to y coordinate Y."""
        return self.tk.getint(self.tk.call(
            self._w, 'nearest', y))

    def scan_mark(self, x, y):
        """Remember the current X, Y coordinates."""
        self.tk.call(self._w, 'scan', 'mark', x, y)

    def scan_dragto(self, x, y):
        """Adjust the view of the listbox to 10 times the
        difference between X and Y and the coordinates given in
        scan_mark."""
        self.tk.call(self._w, 'scan', 'dragto', x, y)

    def see(self, index):
        """Scroll such that INDEX is visible."""
        self.tk.call(self._w, 'see', index)

    def selection_anchor(self, index):
        """Set the fixed end oft the selection to INDEX."""
        self.tk.call(self._w, 'selection', 'anchor', index)

    select_anchor = selection_anchor

    def selection_clear(self, first, last=None):
        """Clear the selection from FIRST to LAST (included)."""
        self.tk.call(self._w,
                 'selection', 'clear', first, last)

    select_clear = selection_clear

    def selection_includes(self, index):
        """Return True if INDEX is part of the selection."""
        return self.tk.getboolean(self.tk.call(
            self._w, 'selection', 'includes', index))

    select_includes = selection_includes

    def selection_set(self, first, last=None):
        """Set the selection from FIRST to LAST (included) without
        changing the currently selected elements."""
        self.tk.call(self._w, 'selection', 'set', first, last)

    select_set = selection_set

    def size(self):
        """Return the number of elements in the listbox."""
        return self.tk.getint(self.tk.call(self._w, 'size'))

    def itemcget(self, index, option):
        """Return the resource value for an ITEM and an OPTION."""
        return self.tk.call(
            (self._w, 'itemcget') + (index, '-'+option))

    def itemconfigure(self, index, cnf=None, **kw):
        """Configure resources of an ITEM.

        The values for resources are specified as keyword arguments.
        To get an overview about the allowed keyword arguments
        call the method without arguments.
        Valid resource names: background, bg, foreground, fg,
        selectbackground, selectforeground."""
        return self._configure(('itemconfigure', index), cnf, kw)

    itemconfig = itemconfigure


class Menu(Widget):
    """Menu widget which allows displaying menu bars, pull-down menus and pop-up menus."""

    def __init__(self, master=None, cnf={}, **kw):
        """Construct menu widget with the parent MASTER.

        Valid resource names: activebackground, activeborderwidth,
        activeforeground, background, bd, bg, borderwidth, cursor,
        disabledforeground, fg, font, foreground, postcommand, relief,
        selectcolor, takefocus, tearoff, tearoffcommand, title, type."""
        Widget.__init__(self, master, 'menu', cnf, kw)

    def tk_popup(self, x, y, entry=""):
        """Post the menu at position X,Y with entry ENTRY."""
        self.tk.call('tk_popup', self._w, x, y, entry)

    def activate(self, index):
        """Activate entry at INDEX."""
        self.tk.call(self._w, 'activate', index)

    def add(self, itemType, cnf={}, **kw):
        """Internal function."""
        self.tk.call((self._w, 'add', itemType) +
                 self._options(cnf, kw))

    def add_cascade(self, cnf={}, **kw):
        """Add hierarchical menu item."""
        self.add('cascade', cnf or kw)

    def add_checkbutton(self, cnf={}, **kw):
        """Add checkbutton menu item."""
        self.add('checkbutton', cnf or kw)

    def add_command(self, cnf={}, **kw):
        """Add command menu item."""
        self.add('command', cnf or kw)

    def add_radiobutton(self, cnf={}, **kw):
        """Add radio menu item."""
        self.add('radiobutton', cnf or kw)

    def add_separator(self, cnf={}, **kw):
        """Add separator."""
        self.add('separator', cnf or kw)

    def insert(self, index, itemType, cnf={}, **kw):
        """Internal function."""
        self.tk.call((self._w, 'insert', index, itemType) +
                 self._options(cnf, kw))

    def insert_cascade(self, index, cnf={}, **kw):
        """Add hierarchical menu item at INDEX."""
        self.insert(index, 'cascade', cnf or kw)

    def insert_checkbutton(self, index, cnf={}, **kw):
        """Add checkbutton menu item at INDEX."""
        self.insert(index, 'checkbutton', cnf or kw)

    def insert_command(self, index, cnf={}, **kw):
        """Add command menu item at INDEX."""
        self.insert(index, 'command', cnf or kw)

    def insert_radiobutton(self, index, cnf={}, **kw):
        """Add radio menu item at INDEX."""
        self.insert(index, 'radiobutton', cnf or kw)

    def insert_separator(self, index, cnf={}, **kw):
        """Add separator at INDEX."""
        self.insert(index, 'separator', cnf or kw)

    def delete(self, index1, index2=None):
        """Delete menu items between INDEX1 and INDEX2 (included)."""
        if index2 is None:
            index2 = index1

        num_index1, num_index2 = self.index(index1), self.index(index2)
        if (num_index1 is None) or (num_index2 is None):
            num_index1, num_index2 = 0, -1

        for i in range(num_index1, num_index2 + 1):
            if 'command' in self.entryconfig(i):
                c = str(self.entrycget(i, 'command'))
                if c:
                    self.deletecommand(c)
        self.tk.call(self._w, 'delete', index1, index2)

    def entrycget(self, index, option):
        """Return the resource value of a menu item for OPTION at INDEX."""
        return self.tk.call(self._w, 'entrycget', index, '-' + option)

    def entryconfigure(self, index, cnf=None, **kw):
        """Configure a menu item at INDEX."""
        return self._configure(('entryconfigure', index), cnf, kw)

    entryconfig = entryconfigure

    def index(self, index):
        """Return the index of a menu item identified by INDEX."""
        i = self.tk.call(self._w, 'index', index)
        return None if i in ('', 'none') else self.tk.getint(i)  # GH-103685.

    def invoke(self, index):
        """Invoke a menu item identified by INDEX and execute
        the associated command."""
        return self.tk.call(self._w, 'invoke', index)

    def post(self, x, y):
        """Display a menu at position X,Y."""
        self.tk.call(self._w, 'post', x, y)

    def type(self, index):
        """Return the type of the menu item at INDEX."""
        return self.tk.call(self._w, 'type', index)

    def unpost(self):
        """Unmap a menu."""
        self.tk.call(self._w, 'unpost')

    def xposition(self, index): # new in Tk 8.5
        """Return the x-position of the leftmost pixel of the menu item
        at INDEX."""
        return self.tk.getint(self.tk.call(self._w, 'xposition', index))

    def yposition(self, index):
        """Return the y-position of the topmost pixel of the menu item at INDEX."""
        return self.tk.getint(self.tk.call(
            self._w, 'yposition', index))


class Menubutton(Widget):
    """Menubutton widget, obsolete since Tk8.0."""

    def __init__(self, master=None, cnf={}, **kw):
        Widget.__init__(self, master, 'menubutton', cnf, kw)


class Message(Widget):
    """Message widget to display multiline text. Obsolete since Label does it too."""

    def __init__(self, master=None, cnf={}, **kw):
        Widget.__init__(self, master, 'message', cnf, kw)


class Radiobutton(Widget):
    """Radiobutton widget which shows only one of several buttons in on-state."""

    def __init__(self, master=None, cnf={}, **kw):
        """Construct a radiobutton widget with the parent MASTER.

        Valid resource names: activebackground, activeforeground, anchor,
        background, bd, bg, bitmap, borderwidth, command, cursor,
        disabledforeground, fg, font, foreground, height,
        highlightbackground, highlightcolor, highlightthickness, image,
        indicatoron, justify, padx, pady, relief, selectcolor, selectimage,
        state, takefocus, text, textvariable, underline, value, variable,
        width, wraplength."""
        Widget.__init__(self, master, 'radiobutton', cnf, kw)

    def deselect(self):
        """Put the button in off-state."""

        self.tk.call(self._w, 'deselect')

    def flash(self):
        """Flash the button."""
        self.tk.call(self._w, 'flash')

    def invoke(self):
        """Toggle the button and invoke a command if given as resource."""
        return self.tk.call(self._w, 'invoke')

    def select(self):
        """Put the button in on-state."""
        self.tk.call(self._w, 'select')


class Scale(Widget):
    """Scale widget which can display a numerical scale."""

    def __init__(self, master=None, cnf={}, **kw):
        """Construct a scale widget with the parent MASTER.

        Valid resource names: activebackground, background, bigincrement, bd,
        bg, borderwidth, command, cursor, digits, fg, font, foreground, from,
        highlightbackground, highlightcolor, highlightthickness, label,
        length, orient, relief, repeatdelay, repeatinterval, resolution,
        showvalue, sliderlength, sliderrelief, state, takefocus,
        tickinterval, to, troughcolor, variable, width."""
        Widget.__init__(self, master, 'scale', cnf, kw)

    def get(self):
        """Get the current value as integer or float."""
        value = self.tk.call(self._w, 'get')
        try:
            return self.tk.getint(value)
        except (ValueError, TypeError, TclError):
            return self.tk.getdouble(value)

    def set(self, value):
        """Set the value to VALUE."""
        self.tk.call(self._w, 'set', value)

    def coords(self, value=None):
        """Return a tuple (X,Y) of the point along the centerline of the
        trough that corresponds to VALUE or the current value if None is
        given."""

        return self._getints(self.tk.call(self._w, 'coords', value))

    def identify(self, x, y):
        """Return where the point X,Y lies. Valid return values are "slider",
        "though1" and "though2"."""
        return self.tk.call(self._w, 'identify', x, y)


class Scrollbar(Widget):
    """Scrollbar widget which displays a slider at a certain position."""

    def __init__(self, master=None, cnf={}, **kw):
        """Construct a scrollbar widget with the parent MASTER.

        Valid resource names: activebackground, activerelief,
        background, bd, bg, borderwidth, command, cursor,
        elementborderwidth, highlightbackground,
        highlightcolor, highlightthickness, jump, orient,
        relief, repeatdelay, repeatinterval, takefocus,
        troughcolor, width."""
        Widget.__init__(self, master, 'scrollbar', cnf, kw)

    def activate(self, index=None):
        """Marks the element indicated by index as active.
        The only index values understood by this method are "arrow1",
        "slider", or "arrow2".  If any other value is specified then no
        element of the scrollbar will be active.  If index is not specified,
        the method returns the name of the element that is currently active,
        or None if no element is active."""
        return self.tk.call(self._w, 'activate', index) or None

    def delta(self, deltax, deltay):
        """Return the fractional change of the scrollbar setting if it
        would be moved by DELTAX or DELTAY pixels."""
        return self.tk.getdouble(
            self.tk.call(self._w, 'delta', deltax, deltay))

    def fraction(self, x, y):
        """Return the fractional value which corresponds to a slider
        position of X,Y."""
        return self.tk.getdouble(self.tk.call(self._w, 'fraction', x, y))

    def identify(self, x, y):
        """Return the element under position X,Y as one of
        "arrow1","slider","arrow2" or ""."""
        return self.tk.call(self._w, 'identify', x, y)

    def get(self):
        """Return the current fractional values (upper and lower end)
        of the slider position."""
        return self._getdoubles(self.tk.call(self._w, 'get'))

    def set(self, first, last):
        """Set the fractional values of the slider position (upper and
        lower ends as value between 0 and 1)."""
        self.tk.call(self._w, 'set', first, last)


class Text(Widget, XView, YView):
    """Text widget which can display text in various forms."""

    def __init__(self, master=None, cnf={}, **kw):
        """Construct a text widget with the parent MASTER.

        STANDARD OPTIONS

            background, borderwidth, cursor,
            exportselection, font, foreground,
            highlightbackground, highlightcolor,
            highlightthickness, insertbackground,
            insertborderwidth, insertofftime,
            insertontime, insertwidth, padx, pady,
            relief, selectbackground,
            selectborderwidth, selectforeground,
            setgrid, takefocus,
            xscrollcommand, yscrollcommand,

        WIDGET-SPECIFIC OPTIONS

            autoseparators, height, maxundo,
            spacing1, spacing2, spacing3,
            state, tabs, undo, width, wrap,

        """
        Widget.__init__(self, master, 'text', cnf, kw)

    def bbox(self, index):
        """Return a tuple of (x,y,width,height) which gives the bounding
        box of the visible part of the character at the given index."""
        return self._getints(
                self.tk.call(self._w, 'bbox', index)) or None

    def compare(self, index1, op, index2):
        """Return whether between index INDEX1 and index INDEX2 the
        relation OP is satisfied. OP is one of <, <=, ==, >=, >, or !=."""
        return self.tk.getboolean(self.tk.call(
            self._w, 'compare', index1, op, index2))

    def count(self, index1, index2, *args): # new in Tk 8.5
        """Counts the number of relevant things between the two indices.
        If index1 is after index2, the result will be a negative number
        (and this holds for each of the possible options).

        The actual items which are counted depends on the options given by
        args. The result is a list of integers, one for the result of each
        counting option given. Valid counting options are "chars",
        "displaychars", "displayindices", "displaylines", "indices",
        "lines", "xpixels" and "ypixels". There is an additional possible
        option "update", which if given then all subsequent options ensure
        that any possible out of date information is recalculated."""
        args = ['-%s' % arg for arg in args]
        args += [index1, index2]
        res = self.tk.call(self._w, 'count', *args) or None
        if res is not None and len(args) <= 3:
            return (res, )
        else:
            return res

    def debug(self, boolean=None):
        """Turn on the internal consistency checks of the B-Tree inside the text
        widget according to BOOLEAN."""
        if boolean is None:
            return self.tk.getboolean(self.tk.call(self._w, 'debug'))
        self.tk.call(self._w, 'debug', boolean)

    def delete(self, index1, index2=None):
        """Delete the characters between INDEX1 and INDEX2 (not included)."""
        self.tk.call(self._w, 'delete', index1, index2)

    def dlineinfo(self, index):
        """Return tuple (x,y,width,height,baseline) giving the bounding box
        and baseline position of the visible part of the line containing
        the character at INDEX."""
        return self._getints(self.tk.call(self._w, 'dlineinfo', index))

    def dump(self, index1, index2=None, command=None, **kw):
        """Return the contents of the widget between index1 and index2.

        The type of contents returned in filtered based on the keyword
        parameters; if 'all', 'image', 'mark', 'tag', 'text', or 'window' are
        given and true, then the corresponding items are returned. The result
        is a list of triples of the form (key, value, index). If none of the
        keywords are true then 'all' is used by default.

        If the 'command' argument is given, it is called once for each element
        of the list of triples, with the values of each triple serving as the
        arguments to the function. In this case the list is not returned."""
        args = []
        func_name = None
        result = None
        if not command:
            # Never call the dump command without the -command flag, since the
            # output could involve Tcl quoting and would be a pain to parse
            # right. Instead just set the command to build a list of triples
            # as if we had done the parsing.
            result = []
            def append_triple(key, value, index, result=result):
                result.append((key, value, index))
            command = append_triple
        try:
            if not isinstance(command, str):
                func_name = command = self._register(command)
            args += ["-command", command]
            for key in kw:
                if kw[key]: args.append("-" + key)
            args.append(index1)
            if index2:
                args.append(index2)
            self.tk.call(self._w, "dump", *args)
            return result
        finally:
            if func_name:
                self.deletecommand(func_name)

    ## new in tk8.4
    def edit(self, *args):
        """Internal method

        This method controls the undo mechanism and
        the modified flag. The exact behavior of the
        command depends on the option argument that
        follows the edit argument. The following forms
        of the command are currently supported:

        edit_modified, edit_redo, edit_reset, edit_separator
        and edit_undo

        """
        return self.tk.call(self._w, 'edit', *args)

    def edit_modified(self, arg=None):
        """Get or Set the modified flag

        If arg is not specified, returns the modified
        flag of the widget. The insert, delete, edit undo and
        edit redo commands or the user can set or clear the
        modified flag. If boolean is specified, sets the
        modified flag of the widget to arg.
        """
        return self.edit("modified", arg)

    def edit_redo(self):
        """Redo the last undone edit

        When the undo option is true, reapplies the last
        undone edits provided no other edits were done since
        then. Generates an error when the redo stack is empty.
        Does nothing when the undo option is false.
        """
        return self.edit("redo")

    def edit_reset(self):
        """Clears the undo and redo stacks
        """
        return self.edit("reset")

    def edit_separator(self):
        """Inserts a separator (boundary) on the undo stack.

        Does nothing when the undo option is false
        """
        return self.edit("separator")

    def edit_undo(self):
        """Undoes the last edit action

        If the undo option is true. An edit action is defined
        as all the insert and delete commands that are recorded
        on the undo stack in between two separators. Generates
        an error when the undo stack is empty. Does nothing
        when the undo option is false
        """
        return self.edit("undo")

    def get(self, index1, index2=None):
        """Return the text from INDEX1 to INDEX2 (not included)."""
        return self.tk.call(self._w, 'get', index1, index2)
    # (Image commands are new in 8.0)

    def image_cget(self, index, option):
        """Return the value of OPTION of an embedded image at INDEX."""
        if option[:1] != "-":
            option = "-" + option
        if option[-1:] == "_":
            option = option[:-1]
        return self.tk.call(self._w, "image", "cget", index, option)

    def image_configure(self, index, cnf=None, **kw):
        """Configure an embedded image at INDEX."""
        return self._configure(('image', 'configure', index), cnf, kw)

    def image_create(self, index, cnf={}, **kw):
        """Create an embedded image at INDEX."""
        return self.tk.call(
                 self._w, "image", "create", index,
                 *self._options(cnf, kw))

    def image_names(self):
        """Return all names of embedded images in this widget."""
        return self.tk.call(self._w, "image", "names")

    def index(self, index):
        """Return the index in the form line.char for INDEX."""
        return str(self.tk.call(self._w, 'index', index))

    def insert(self, index, chars, *args):
        """Insert CHARS before the characters at INDEX. An additional
        tag can be given in ARGS. Additional CHARS and tags can follow in ARGS."""
        self.tk.call((self._w, 'insert', index, chars) + args)

    def mark_gravity(self, markName, direction=None):
        """Change the gravity of a mark MARKNAME to DIRECTION (LEFT or RIGHT).
        Return the current value if None is given for DIRECTION."""
        return self.tk.call(
            (self._w, 'mark', 'gravity', markName, direction))

    def mark_names(self):
        """Return all mark names."""
        return self.tk.splitlist(self.tk.call(
            self._w, 'mark', 'names'))

    def mark_set(self, markName, index):
        """Set mark MARKNAME before the character at INDEX."""
        self.tk.call(self._w, 'mark', 'set', markName, index)

    def mark_unset(self, *markNames):
        """Delete all marks in MARKNAMES."""
        self.tk.call((self._w, 'mark', 'unset') + markNames)

    def mark_next(self, index):
        """Return the name of the next mark after INDEX."""
        return self.tk.call(self._w, 'mark', 'next', index) or None

    def mark_previous(self, index):
        """Return the name of the previous mark before INDEX."""
        return self.tk.call(self._w, 'mark', 'previous', index) or None

    def peer_create(self, newPathName, cnf={}, **kw): # new in Tk 8.5
        """Creates a peer text widget with the given newPathName, and any
        optional standard configuration options. By default the peer will
        have the same start and end line as the parent widget, but
        these can be overridden with the standard configuration options."""
        self.tk.call(self._w, 'peer', 'create', newPathName,
            *self._options(cnf, kw))

    def peer_names(self): # new in Tk 8.5
        """Returns a list of peers of this widget (this does not include
        the widget itself)."""
        return self.tk.splitlist(self.tk.call(self._w, 'peer', 'names'))

    def replace(self, index1, index2, chars, *args): # new in Tk 8.5
        """Replaces the range of characters between index1 and index2 with
        the given characters and tags specified by args.

        See the method insert for some more information about args, and the
        method delete for information about the indices."""
        self.tk.call(self._w, 'replace', index1, index2, chars, *args)

    def scan_mark(self, x, y):
        """Remember the current X, Y coordinates."""
        self.tk.call(self._w, 'scan', 'mark', x, y)

    def scan_dragto(self, x, y):
        """Adjust the view of the text to 10 times the
        difference between X and Y and the coordinates given in
        scan_mark."""
        self.tk.call(self._w, 'scan', 'dragto', x, y)

    def search(self, pattern, index, stopindex=None,
           forwards=None, backwards=None, exact=None,
           regexp=None, nocase=None, count=None, elide=None):
        """Search PATTERN beginning from INDEX until STOPINDEX.
        Return the index of the first character of a match or an
        empty string."""
        args = [self._w, 'search']
        if forwards: args.append('-forwards')
        if backwards: args.append('-backwards')
        if exact: args.append('-exact')
        if regexp: args.append('-regexp')
        if nocase: args.append('-nocase')
        if elide: args.append('-elide')
        if count: args.append('-count'); args.append(count)
        if pattern and pattern[0] == '-': args.append('--')
        args.append(pattern)
        args.append(index)
        if stopindex: args.append(stopindex)
        return str(self.tk.call(tuple(args)))

    def see(self, index):
        """Scroll such that the character at INDEX is visible."""
        self.tk.call(self._w, 'see', index)

    def tag_add(self, tagName, index1, *args):
        """Add tag TAGNAME to all characters between INDEX1 and index2 in ARGS.
        Additional pairs of indices may follow in ARGS."""
        self.tk.call(
            (self._w, 'tag', 'add', tagName, index1) + args)

    def tag_unbind(self, tagName, sequence, funcid=None):
        """Unbind for all characters with TAGNAME for event SEQUENCE  the
        function identified with FUNCID."""
        return self._unbind((self._w, 'tag', 'bind', tagName, sequence), funcid)

    def tag_bind(self, tagName, sequence, func, add=None):
        """Bind to all characters with TAGNAME at event SEQUENCE a call to function FUNC.

        An additional boolean parameter ADD specifies whether FUNC will be
        called additionally to the other bound function or whether it will
        replace the previous function. See bind for the return value."""
        return self._bind((self._w, 'tag', 'bind', tagName),
                  sequence, func, add)

    def _tag_bind(self, tagName, sequence=None, func=None, add=None):
        # For tests only
        return self._bind((self._w, 'tag', 'bind', tagName),
                  sequence, func, add)

    def tag_cget(self, tagName, option):
        """Return the value of OPTION for tag TAGNAME."""
        if option[:1] != '-':
            option = '-' + option
        if option[-1:] == '_':
            option = option[:-1]
        return self.tk.call(self._w, 'tag', 'cget', tagName, option)

    def tag_configure(self, tagName, cnf=None, **kw):
        """Configure a tag TAGNAME."""
        return self._configure(('tag', 'configure', tagName), cnf, kw)

    tag_config = tag_configure

    def tag_delete(self, *tagNames):
        """Delete all tags in TAGNAMES."""
        self.tk.call((self._w, 'tag', 'delete') + tagNames)

    def tag_lower(self, tagName, belowThis=None):
        """Change the priority of tag TAGNAME such that it is lower
        than the priority of BELOWTHIS."""
        self.tk.call(self._w, 'tag', 'lower', tagName, belowThis)

    def tag_names(self, index=None):
        """Return a list of all tag names."""
        return self.tk.splitlist(
            self.tk.call(self._w, 'tag', 'names', index))

    def tag_nextrange(self, tagName, index1, index2=None):
        """Return a list of start and end index for the first sequence of
        characters between INDEX1 and INDEX2 which all have tag TAGNAME.
        The text is searched forward from INDEX1."""
        return self.tk.splitlist(self.tk.call(
            self._w, 'tag', 'nextrange', tagName, index1, index2))

    def tag_prevrange(self, tagName, index1, index2=None):
        """Return a list of start and end index for the first sequence of
        characters between INDEX1 and INDEX2 which all have tag TAGNAME.
        The text is searched backwards from INDEX1."""
        return self.tk.splitlist(self.tk.call(
            self._w, 'tag', 'prevrange', tagName, index1, index2))

    def tag_raise(self, tagName, aboveThis=None):
        """Change the priority of tag TAGNAME such that it is higher
        than the priority of ABOVETHIS."""
        self.tk.call(
            self._w, 'tag', 'raise', tagName, aboveThis)

    def tag_ranges(self, tagName):
        """Return a list of ranges of text which have tag TAGNAME."""
        return self.tk.splitlist(self.tk.call(
            self._w, 'tag', 'ranges', tagName))

    def tag_remove(self, tagName, index1, index2=None):
        """Remove tag TAGNAME from all characters between INDEX1 and INDEX2."""
        self.tk.call(
            self._w, 'tag', 'remove', tagName, index1, index2)

    def window_cget(self, index, option):
        """Return the value of OPTION of an embedded window at INDEX."""
        if option[:1] != '-':
            option = '-' + option
        if option[-1:] == '_':
            option = option[:-1]
        return self.tk.call(self._w, 'window', 'cget', index, option)

    def window_configure(self, index, cnf=None, **kw):
        """Configure an embedded window at INDEX."""
        return self._configure(('window', 'configure', index), cnf, kw)

    window_config = window_configure

    def window_create(self, index, cnf={}, **kw):
        """Create a window at INDEX."""
        self.tk.call(
              (self._w, 'window', 'create', index)
              + self._options(cnf, kw))

    def window_names(self):
        """Return all names of embedded windows in this widget."""
        return self.tk.splitlist(
            self.tk.call(self._w, 'window', 'names'))

    def yview_pickplace(self, *what):
        """Obsolete function, use see."""
        self.tk.call((self._w, 'yview', '-pickplace') + what)


class _setit:
    """Internal class. It wraps the command in the widget OptionMenu."""

    def __init__(self, var, value, callback=None):
        self.__value = value
        self.__var = var
        self.__callback = callback

    def __call__(self, *args):
        self.__var.set(self.__value)
        if self.__callback is not None:
            self.__callback(self.__value, *args)


class OptionMenu(Menubutton):
    """OptionMenu which allows the user to select a value from a menu."""

    def __init__(self, master, variable, value, *values, **kwargs):
        """Construct an optionmenu widget with the parent MASTER, with
        the resource textvariable set to VARIABLE, the initially selected
        value VALUE, the other menu values VALUES and an additional
        keyword argument command."""
        kw = {"borderwidth": 2, "textvariable": variable,
              "indicatoron": 1, "relief": RAISED, "anchor": "c",
              "highlightthickness": 2}
        Widget.__init__(self, master, "menubutton", kw)
        self.widgetName = 'tk_optionMenu'
        menu = self.__menu = Menu(self, name="menu", tearoff=0)
        self.menuname = menu._w
        # 'command' is the only supported keyword
        callback = kwargs.get('command')
        if 'command' in kwargs:
            del kwargs['command']
        if kwargs:
            raise TclError('unknown option -'+next(iter(kwargs)))
        menu.add_command(label=value,
                 command=_setit(variable, value, callback))
        for v in values:
            menu.add_command(label=v,
                     command=_setit(variable, v, callback))
        self["menu"] = menu

    def __getitem__(self, name):
        if name == 'menu':
            return self.__menu
        return Widget.__getitem__(self, name)

    def destroy(self):
        """Destroy this widget and the associated menu."""
        Menubutton.destroy(self)
        self.__menu = None


class Image:
    """Base class for images."""
    _last_id = 0

    def __init__(self, imgtype, name=None, cnf={}, master=None, **kw):
        self.name = None
        if master is None:
            master = _get_default_root('create image')
        self.tk = getattr(master, 'tk', master)
        if not name:
            Image._last_id += 1
            name = "pyimage%r" % (Image._last_id,) # tk itself would use image<x>
        if kw and cnf: cnf = _cnfmerge((cnf, kw))
        elif kw: cnf = kw
        options = ()
        for k, v in cnf.items():
            options = options + ('-'+k, v)
        self.tk.call(('image', 'create', imgtype, name,) + options)
        self.name = name

    def __str__(self): return self.name

    def __del__(self):
        if self.name:
            try:
                self.tk.call('image', 'delete', self.name)
            except TclError:
                # May happen if the root was destroyed
                pass

    def __setitem__(self, key, value):
        self.tk.call(self.name, 'configure', '-'+key, value)

    def __getitem__(self, key):
        return self.tk.call(self.name, 'configure', '-'+key)

    def configure(self, **kw):
        """Configure the image."""
        res = ()
        for k, v in _cnfmerge(kw).items():
            if v is not None:
                if k[-1] == '_': k = k[:-1]
                res = res + ('-'+k, v)
        self.tk.call((self.name, 'config') + res)

    config = configure

    def height(self):
        """Return the height of the image."""
        return self.tk.getint(
            self.tk.call('image', 'height', self.name))

    def type(self):
        """Return the type of the image, e.g. "photo" or "bitmap"."""
        return self.tk.call('image', 'type', self.name)

    def width(self):
        """Return the width of the image."""
        return self.tk.getint(
            self.tk.call('image', 'width', self.name))


class PhotoImage(Image):
    """Widget which can display images in PGM, PPM, GIF, PNG format."""

    def __init__(self, name=None, cnf={}, master=None, **kw):
        """Create an image with NAME.

        Valid resource names: data, format, file, gamma, height, palette,
        width."""
        Image.__init__(self, 'photo', name, cnf, master, **kw)

    def blank(self):
        """Display a transparent image."""
        self.tk.call(self.name, 'blank')

    def cget(self, option):
        """Return the value of OPTION."""
        return self.tk.call(self.name, 'cget', '-' + option)
    # XXX config

    def __getitem__(self, key):
        return self.tk.call(self.name, 'cget', '-' + key)
    # XXX copy -from, -to, ...?

    def copy(self):
        """Return a new PhotoImage with the same image as this widget."""
        destImage = PhotoImage(master=self.tk)
        self.tk.call(destImage, 'copy', self.name)
        return destImage

    def zoom(self, x, y=''):
        """Return a new PhotoImage with the same image as this widget
        but zoom it with a factor of x in the X direction and y in the Y
        direction.  If y is not given, the default value is the same as x.
        """
        destImage = PhotoImage(master=self.tk)
        if y=='': y=x
        self.tk.call(destImage, 'copy', self.name, '-zoom',x,y)
        return destImage

    def subsample(self, x, y=''):
        """Return a new PhotoImage based on the same image as this widget
        but use only every Xth or Yth pixel.  If y is not given, the
        default value is the same as x.
        """
        destImage = PhotoImage(master=self.tk)
        if y=='': y=x
        self.tk.call(destImage, 'copy', self.name, '-subsample',x,y)
        return destImage

    def get(self, x, y):
        """Return the color (red, green, blue) of the pixel at X,Y."""
        return self.tk.call(self.name, 'get', x, y)

    def put(self, data, to=None):
        """Put row formatted colors to image starting from
        position TO, e.g. image.put("{red green} {blue yellow}", to=(4,6))"""
        args = (self.name, 'put', data)
        if to:
            if to[0] == '-to':
                to = to[1:]
            args = args + ('-to',) + tuple(to)
        self.tk.call(args)
    # XXX read

    def write(self, filename, format=None, from_coords=None):
        """Write image to file FILENAME in FORMAT starting from
        position FROM_COORDS."""
        args = (self.name, 'write', filename)
        if format:
            args = args + ('-format', format)
        if from_coords:
            args = args + ('-from',) + tuple(from_coords)
        self.tk.call(args)

    def transparency_get(self, x, y):
        """Return True if the pixel at x,y is transparent."""
        return self.tk.getboolean(self.tk.call(
            self.name, 'transparency', 'get', x, y))

    def transparency_set(self, x, y, boolean):
        """Set the transparency of the pixel at x,y."""
        self.tk.call(self.name, 'transparency', 'set', x, y, boolean)


class BitmapImage(Image):
    """Widget which can display images in XBM format."""

    def __init__(self, name=None, cnf={}, master=None, **kw):
        """Create a bitmap with NAME.

        Valid resource names: background, data, file, foreground, maskdata, maskfile."""
        Image.__init__(self, 'bitmap', name, cnf, master, **kw)


def image_names():
    tk = _get_default_root('use image_names()').tk
    return tk.splitlist(tk.call('image', 'names'))


def image_types():
    tk = _get_default_root('use image_types()').tk
    return tk.splitlist(tk.call('image', 'types'))


class Spinbox(Widget, XView):
    """spinbox widget."""

    def __init__(self, master=None, cnf={}, **kw):
        """Construct a spinbox widget with the parent MASTER.

        STANDARD OPTIONS

            activebackground, background, borderwidth,
            cursor, exportselection, font, foreground,
            highlightbackground, highlightcolor,
            highlightthickness, insertbackground,
            insertborderwidth, insertofftime,
            insertontime, insertwidth, justify, relief,
            repeatdelay, repeatinterval,
            selectbackground, selectborderwidth
            selectforeground, takefocus, textvariable
            xscrollcommand.

        WIDGET-SPECIFIC OPTIONS

            buttonbackground, buttoncursor,
            buttondownrelief, buttonuprelief,
            command, disabledbackground,
            disabledforeground, format, from,
            invalidcommand, increment,
            readonlybackground, state, to,
            validate, validatecommand values,
            width, wrap,
        """
        Widget.__init__(self, master, 'spinbox', cnf, kw)

    def bbox(self, index):
        """Return a tuple of X1,Y1,X2,Y2 coordinates for a
        rectangle which encloses the character given by index.

        The first two elements of the list give the x and y
        coordinates of the upper-left corner of the screen
        area covered by the character (in pixels relative
        to the widget) and the last two elements give the
        width and height of the character, in pixels. The
        bounding box may refer to a region outside the
        visible area of the window.
        """
        return self._getints(self.tk.call(self._w, 'bbox', index)) or None

    def delete(self, first, last=None):
        """Delete one or more elements of the spinbox.

        First is the index of the first character to delete,
        and last is the index of the character just after
        the last one to delete. If last isn't specified it
        defaults to first+1, i.e. a single character is
        deleted.  This command returns an empty string.
        """
        return self.tk.call(self._w, 'delete', first, last)

    def get(self):
        """Returns the spinbox's string"""
        return self.tk.call(self._w, 'get')

    def icursor(self, index):
        """Alter the position of the insertion cursor.

        The insertion cursor will be displayed just before
        the character given by index. Returns an empty string
        """
        return self.tk.call(self._w, 'icursor', index)

    def identify(self, x, y):
        """Returns the name of the widget at position x, y

        Return value is one of: none, buttondown, buttonup, entry
        """
        return self.tk.call(self._w, 'identify', x, y)

    def index(self, index):
        """Returns the numerical index corresponding to index
        """
        return self.tk.call(self._w, 'index', index)

    def insert(self, index, s):
        """Insert string s at index

         Returns an empty string.
        """
        return self.tk.call(self._w, 'insert', index, s)

    def invoke(self, element):
        """Causes the specified element to be invoked

        The element could be buttondown or buttonup
        triggering the action associated with it.
        """
        return self.tk.call(self._w, 'invoke', element)

    def scan(self, *args):
        """Internal function."""
        return self._getints(
            self.tk.call((self._w, 'scan') + args)) or ()

    def scan_mark(self, x):
        """Records x and the current view in the spinbox window;

        used in conjunction with later scan dragto commands.
        Typically this command is associated with a mouse button
        press in the widget. It returns an empty string.
        """
        return self.scan("mark", x)

    def scan_dragto(self, x):
        """Compute the difference between the given x argument
        and the x argument to the last scan mark command

        It then adjusts the view left or right by 10 times the
        difference in x-coordinates. This command is typically
        associated with mouse motion events in the widget, to
        produce the effect of dragging the spinbox at high speed
        through the window. The return value is an empty string.
        """
        return self.scan("dragto", x)

    def selection(self, *args):
        """Internal function."""
        return self._getints(
            self.tk.call((self._w, 'selection') + args)) or ()

    def selection_adjust(self, index):
        """Locate the end of the selection nearest to the character
        given by index,

        Then adjust that end of the selection to be at index
        (i.e including but not going beyond index). The other
        end of the selection is made the anchor point for future
        select to commands. If the selection isn't currently in
        the spinbox, then a new selection is created to include
        the characters between index and the most recent selection
        anchor point, inclusive.
        """
        return self.selection("adjust", index)

    def selection_clear(self):
        """Clear the selection

        If the selection isn't in this widget then the
        command has no effect.
        """
        return self.selection("clear")

    def selection_element(self, element=None):
        """Sets or gets the currently selected element.

        If a spinbutton element is specified, it will be
        displayed depressed.
        """
        return self.tk.call(self._w, 'selection', 'element', element)

    def selection_from(self, index):
        """Set the fixed end of a selection to INDEX."""
        self.selection('from', index)

    def selection_present(self):
        """Return True if there are characters selected in the spinbox, False
        otherwise."""
        return self.tk.getboolean(
            self.tk.call(self._w, 'selection', 'present'))

    def selection_range(self, start, end):
        """Set the selection from START to END (not included)."""
        self.selection('range', start, end)

    def selection_to(self, index):
        """Set the variable end of a selection to INDEX."""
        self.selection('to', index)

###########################################################################


class LabelFrame(Widget):
    """labelframe widget."""

    def __init__(self, master=None, cnf={}, **kw):
        """Construct a labelframe widget with the parent MASTER.

        STANDARD OPTIONS

            borderwidth, cursor, font, foreground,
            highlightbackground, highlightcolor,
            highlightthickness, padx, pady, relief,
            takefocus, text

        WIDGET-SPECIFIC OPTIONS

            background, class, colormap, container,
            height, labelanchor, labelwidget,
            visual, width
        """
        Widget.__init__(self, master, 'labelframe', cnf, kw)

########################################################################


class PanedWindow(Widget):
    """panedwindow widget."""

    def __init__(self, master=None, cnf={}, **kw):
        """Construct a panedwindow widget with the parent MASTER.

        STANDARD OPTIONS

            background, borderwidth, cursor, height,
            orient, relief, width

        WIDGET-SPECIFIC OPTIONS

            handlepad, handlesize, opaqueresize,
            sashcursor, sashpad, sashrelief,
            sashwidth, showhandle,
        """
        Widget.__init__(self, master, 'panedwindow', cnf, kw)

    def add(self, child, **kw):
        """Add a child widget to the panedwindow in a new pane.

        The child argument is the name of the child widget
        followed by pairs of arguments that specify how to
        manage the windows. The possible options and values
        are the ones accepted by the paneconfigure method.
        """
        self.tk.call((self._w, 'add', child) + self._options(kw))

    def remove(self, child):
        """Remove the pane containing child from the panedwindow

        All geometry management options for child will be forgotten.
        """
        self.tk.call(self._w, 'forget', child)

    forget = remove

    def identify(self, x, y):
        """Identify the panedwindow component at point x, y

        If the point is over a sash or a sash handle, the result
        is a two element list containing the index of the sash or
        handle, and a word indicating whether it is over a sash
        or a handle, such as {0 sash} or {2 handle}. If the point
        is over any other part of the panedwindow, the result is
        an empty list.
        """
        return self.tk.call(self._w, 'identify', x, y)

    def proxy(self, *args):
        """Internal function."""
        return self._getints(
            self.tk.call((self._w, 'proxy') + args)) or ()

    def proxy_coord(self):
        """Return the x and y pair of the most recent proxy location
        """
        return self.proxy("coord")

    def proxy_forget(self):
        """Remove the proxy from the display.
        """
        return self.proxy("forget")

    def proxy_place(self, x, y):
        """Place the proxy at the given x and y coordinates.
        """
        return self.proxy("place", x, y)

    def sash(self, *args):
        """Internal function."""
        return self._getints(
            self.tk.call((self._w, 'sash') + args)) or ()

    def sash_coord(self, index):
        """Return the current x and y pair for the sash given by index.

        Index must be an integer between 0 and 1 less than the
        number of panes in the panedwindow. The coordinates given are
        those of the top left corner of the region containing the sash.
        pathName sash dragto index x y This command computes the
        difference between the given coordinates and the coordinates
        given to the last sash coord command for the given sash. It then
        moves that sash the computed difference. The return value is the
        empty string.
        """
        return self.sash("coord", index)

    def sash_mark(self, index):
        """Records x and y for the sash given by index;

        Used in conjunction with later dragto commands to move the sash.
        """
        return self.sash("mark", index)

    def sash_place(self, index, x, y):
        """Place the sash given by index at the given coordinates
        """
        return self.sash("place", index, x, y)

    def panecget(self, child, option):
        """Query a management option for window.

        Option may be any value allowed by the paneconfigure subcommand
        """
        return self.tk.call(
            (self._w, 'panecget') + (child, '-'+option))

    def paneconfigure(self, tagOrId, cnf=None, **kw):
        """Query or modify the management options for window.

        If no option is specified, returns a list describing all
        of the available options for pathName.  If option is
        specified with no value, then the command returns a list
        describing the one named option (this list will be identical
        to the corresponding sublist of the value returned if no
        option is specified). If one or more option-value pairs are
        specified, then the command modifies the given widget
        option(s) to have the given value(s); in this case the
        command returns an empty string. The following options
        are supported:

        after window
            Insert the window after the window specified. window
            should be the name of a window already managed by pathName.
        before window
            Insert the window before the window specified. window
            should be the name of a window already managed by pathName.
        height size
            Specify a height for the window. The height will be the
            outer dimension of the window including its border, if
            any. If size is an empty string, or if -height is not
            specified, then the height requested internally by the
            window will be used initially; the height may later be
            adjusted by the movement of sashes in the panedwindow.
            Size may be any value accepted by Tk_GetPixels.
        minsize n
            Specifies that the size of the window cannot be made
            less than n. This constraint only affects the size of
            the widget in the paned dimension -- the x dimension
            for horizontal panedwindows, the y dimension for
            vertical panedwindows. May be any value accepted by
            Tk_GetPixels.
        padx n
            Specifies a non-negative value indicating how much
            extra space to leave on each side of the window in
            the X-direction. The value may have any of the forms
            accepted by Tk_GetPixels.
        pady n
            Specifies a non-negative value indicating how much
            extra space to leave on each side of the window in
            the Y-direction. The value may have any of the forms
            accepted by Tk_GetPixels.
        sticky style
            If a window's pane is larger than the requested
            dimensions of the window, this option may be used
            to position (or stretch) the window within its pane.
            Style is a string that contains zero or more of the
            characters n, s, e or w. The string can optionally
            contains spaces or commas, but they are ignored. Each
            letter refers to a side (north, south, east, or west)
            that the window will "stick" to. If both n and s
            (or e and w) are specified, the window will be
            stretched to fill the entire height (or width) of
            its cavity.
        width size
            Specify a width for the window. The width will be
            the outer dimension of the window including its
            border, if any. If size is an empty string, or
            if -width is not specified, then the width requested
            internally by the window will be used initially; the
            width may later be adjusted by the movement of sashes
            in the panedwindow. Size may be any value accepted by
            Tk_GetPixels.

        """
        if cnf is None and not kw:
            return self._getconfigure(self._w, 'paneconfigure', tagOrId)
        if isinstance(cnf, str) and not kw:
            return self._getconfigure1(
                self._w, 'paneconfigure', tagOrId, '-'+cnf)
        self.tk.call((self._w, 'paneconfigure', tagOrId) +
                 self._options(cnf, kw))

    paneconfig = paneconfigure

    def panes(self):
        """Returns an ordered list of the child panes."""
        return self.tk.splitlist(self.tk.call(self._w, 'panes'))

# Test:


def _test():
    root = Tk()
    text = "This is Tcl/Tk %s" % root.globalgetvar('tk_patchLevel')
    text += "\nThis should be a cedilla: \xe7"
    label = Label(root, text=text)
    label.pack()
    test = Button(root, text="Click me!",
              command=lambda root=root: root.test.configure(
                  text="[%s]" % root.test['text']))
    test.pack()
    root.test = test
    quit = Button(root, text="QUIT", command=root.destroy)
    quit.pack()
    # The following three commands are needed so the window pops
    # up on top on Windows...
    root.iconify()
    root.update()
    root.deiconify()
    root.mainloop()


__all__ = [name for name, obj in globals().items()
           if not name.startswith('_') and not isinstance(obj, types.ModuleType)
           and name not in {'wantobjects'}]

if __name__ == '__main__':
    _test()
