"""Generic interface to all dbm clones.

Use

        import dbm
        d = dbm.open(file, 'w', 0o666)

The returned object is a dbm.gnu, dbm.ndbm or dbm.dumb object, dependent on the
type of database being opened (determined by the whichdb function) in the case
of an existing dbm. If the dbm does not exist and the create or new flag ('c'
or 'n') was specified, the dbm type will be determined by the availability of
the modules (tested in the above order).

It has the following interface (key and data are strings):

        d[key] = data   # store data at key (may override data at
                        # existing key)
        data = d[key]   # retrieve data at key (raise KeyError if no
                        # such key)
        del d[key]      # delete data stored at key (raises KeyError
                        # if no such key)
        flag = key in d # true if the key exists
        list = d.keys() # return a list of all existing keys (slow!)

Future versions may change the order in which implementations are
tested for existence, and add interfaces to other dbm-like
implementations.
"""

__all__ = ['open', 'whichdb', 'error']

import io
import os
import struct
import sys


class error(Exception):
    pass

_names = ['dbm.gnu', 'dbm.ndbm', 'dbm.dumb']
_defaultmod = None
_modules = {}

error = (error, OSError)

try:
    from dbm import ndbm
except ImportError:
    ndbm = None


def open(file, flag='r', mode=0o666):
    """Open or create database at path given by *file*.

    Optional argument *flag* can be 'r' (default) for read-only access, 'w'
    for read-write access of an existing database, 'c' for read-write access
    to a new or existing database, and 'n' for read-write access to a new
    database.

    Note: 'r' and 'w' fail if the database doesn't exist; 'c' creates it
    only if it doesn't exist; and 'n' always creates a new database.
    """
    global _defaultmod
    if _defaultmod is None:
        for name in _names:
            try:
                mod = __import__(name, fromlist=['open'])
            except ImportError:
                continue
            if not _defaultmod:
                _defaultmod = mod
            _modules[name] = mod
        if not _defaultmod:
            raise ImportError("no dbm clone found; tried %s" % _names)

    # guess the type of an existing database, if not creating a new one
    result = whichdb(file) if 'n' not in flag else None
    if result is None:
        # db doesn't exist or 'n' flag was specified to create a new db
        if 'c' in flag or 'n' in flag:
            # file doesn't exist and the new flag was used so use default type
            mod = _defaultmod
        else:
            raise error[0]("db file doesn't exist; "
                           "use 'c' or 'n' flag to create a new db")
    elif result == "":
        # db type cannot be determined
        raise error[0]("db type could not be determined")
    elif result not in _modules:
        raise error[0]("db type is {0}, but the module is not "
                       "available".format(result))
    else:
        mod = _modules[result]
    return mod.open(file, flag, mode)


def whichdb(filename):
    """Guess which db package to use to open a db file.

    Return values:

    - None if the database file can't be read;
    - empty string if the file can be read but can't be recognized
    - the name of the dbm submodule (e.g. "ndbm" or "gnu") if recognized.

    Importing the given module may still fail, and opening the
    database using that module may still fail.
    """

    # Check for ndbm first -- this has a .pag and a .dir file
    filename = os.fsencode(filename)
    try:
        f = io.open(filename + b".pag", "rb")
        f.close()
        f = io.open(filename + b".dir", "rb")
        f.close()
        return "dbm.ndbm"
    except OSError:
        # some dbm emulations based on Berkeley DB generate a .db file
        # some do not, but they should be caught by the bsd checks
        try:
            f = io.open(filename + b".db", "rb")
            f.close()
            # guarantee we can actually open the file using dbm
            # kind of overkill, but since we are dealing with emulations
            # it seems like a prudent step
            if ndbm is not None:
                d = ndbm.open(filename)
                d.close()
                return "dbm.ndbm"
        except OSError:
            pass

    # Check for dumbdbm next -- this has a .dir and a .dat file
    try:
        # First check for presence of files
        os.stat(filename + b".dat")
        size = os.stat(filename + b".dir").st_size
        # dumbdbm files with no keys are empty
        if size == 0:
            return "dbm.dumb"
        f = io.open(filename + b".dir", "rb")
        try:
            if f.read(1) in (b"'", b'"'):
                return "dbm.dumb"
        finally:
            f.close()
    except OSError:
        pass

    # See if the file exists, return None if not
    try:
        f = io.open(filename, "rb")
    except OSError:
        return None

    with f:
        # Read the start of the file -- the magic number
        s16 = f.read(16)
    s = s16[0:4]

    # Return "" if not at least 4 bytes
    if len(s) != 4:
        return ""

    # Convert to 4-byte int in native byte order -- return "" if impossible
    try:
        (magic,) = struct.unpack("=l", s)
    except struct.error:
        return ""

    # Check for GNU dbm
    if magic in (0x13579ace, 0x13579acd, 0x13579acf):
        return "dbm.gnu"

    # Later versions of Berkeley db hash file have a 12-byte pad in
    # front of the file type
    try:
        (magic,) = struct.unpack("=l", s16[-4:])
    except struct.error:
        return ""

    # Unknown
    return ""


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        print(whichdb(filename) or "UNKNOWN", filename)
