import os.path
import sys


# Enable running IDLE with idlelib in a non-standard location.
# This was once used to run development versions of IDLE.
# Because PEP 434 declared idle.py a public interface,
# removal should require deprecation.
idlelib_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if idlelib_dir not in sys.path:
    sys.path.insert(0, idlelib_dir)

from idlelib.pyshell import main  # This is subject to change
main()
