try:
    import idlelib.pyshell
except ImportError:
    # IDLE is not installed, but maybe pyshell is on sys.path:
    from . import pyshell
    import os
    idledir = os.path.dirname(os.path.abspath(pyshell.__file__))
    if idledir != os.getcwd():
        # We're not in the IDLE directory, help the subprocess find run.py
        pypath = os.environ.get('PYTHONPATH', '')
        if pypath:
            os.environ['PYTHONPATH'] = pypath + ':' + idledir
        else:
            os.environ['PYTHONPATH'] = idledir
    pyshell.main()
else:
    idlelib.pyshell.main()
