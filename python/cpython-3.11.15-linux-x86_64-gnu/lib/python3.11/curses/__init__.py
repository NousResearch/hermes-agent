"""curses

The main package for curses support for Python.  Normally used by importing
the package, and perhaps a particular module inside it.

   import curses
   from curses import textpad
   curses.initscr()
   ...

"""

from _curses import *
import os as _os
import sys as _sys

# Some constants, most notably the ACS_* ones, are only added to the C
# _curses module's dictionary after initscr() is called.  (Some
# versions of SGI's curses don't define values for those constants
# until initscr() has been called.)  This wrapper function calls the
# underlying C initscr(), and then copies the constants from the
# _curses module to the curses package's dictionary.  Don't do 'from
# curses import *' if you'll be needing the ACS_* constants.

def initscr():
    import _curses, curses
    # we call setupterm() here because it raises an error
    # instead of calling exit() in error cases.
    setupterm(term=_os.environ.get("TERM", "unknown"),
              fd=_sys.__stdout__.fileno())
    stdscr = _curses.initscr()
    for key, value in _curses.__dict__.items():
        if key[0:4] == 'ACS_' or key in ('LINES', 'COLS'):
            setattr(curses, key, value)

    return stdscr

# This is a similar wrapper for start_color(), which adds the COLORS and
# COLOR_PAIRS variables which are only available after start_color() is
# called.

def start_color():
    import _curses, curses
    retval = _curses.start_color()
    if hasattr(_curses, 'COLORS'):
        curses.COLORS = _curses.COLORS
    if hasattr(_curses, 'COLOR_PAIRS'):
        curses.COLOR_PAIRS = _curses.COLOR_PAIRS
    return retval

# Import Python has_key() implementation if _curses doesn't contain has_key()

try:
    has_key
except NameError:
    from .has_key import has_key

# Wrapper for the entire curses-based application.  Runs a function which
# should be the rest of your curses-based application.  If the application
# raises an exception, wrapper() will restore the terminal to a sane state so
# you can read the resulting traceback.

def wrapper(func, /, *args, **kwds):
    """Wrapper function that initializes curses and calls another function,
    restoring normal keyboard/screen behavior on error.
    The callable object 'func' is then passed the main window 'stdscr'
    as its first argument, followed by any other arguments passed to
    wrapper().
    """

    try:
        # Initialize curses
        stdscr = initscr()

        # Turn off echoing of keys, and enter cbreak mode,
        # where no buffering is performed on keyboard input
        noecho()
        cbreak()

        # In keypad mode, escape sequences for special keys
        # (like the cursor keys) will be interpreted and
        # a special value like curses.KEY_LEFT will be returned
        stdscr.keypad(1)

        # Start color, too.  Harmless if the terminal doesn't have
        # color; user can test with has_color() later on.  The try/catch
        # works around a minor bit of over-conscientiousness in the curses
        # module -- the error return from C start_color() is ignorable.
        try:
            start_color()
        except:
            pass

        return func(stdscr, *args, **kwds)
    finally:
        # Set everything back to normal
        if 'stdscr' in locals():
            stdscr.keypad(0)
            echo()
            nocbreak()
            endwin()
