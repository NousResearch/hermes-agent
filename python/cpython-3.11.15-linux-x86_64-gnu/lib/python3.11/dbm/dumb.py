"""A dumb and slow but simple dbm clone.

For database spam, spam.dir contains the index (a text file),
spam.bak *may* contain a backup of the index (also a text file),
while spam.dat contains the data (a binary file).

XXX TO DO:

- seems to contain a bug when updating...

- reclaim free space (currently, space once occupied by deleted or expanded
items is never reused)

- support concurrent access (currently, if two processes take turns making
updates, they can mess up the index)

- support efficient access to large databases (currently, the whole index
is read when the database is opened, and some updates rewrite the whole index)

- support opening for read-only (flag = 'm')

"""

import ast as _ast
import io as _io
import os as _os
import collections.abc

__all__ = ["error", "open"]

_BLOCKSIZE = 512

error = OSError

class _Database(collections.abc.MutableMapping):

    # The on-disk directory and data files can remain in mutually
    # inconsistent states for an arbitrarily long time (see comments
    # at the end of __setitem__).  This is only repaired when _commit()
    # gets called.  One place _commit() gets called is from __del__(),
    # and if that occurs at program shutdown time, module globals may
    # already have gotten rebound to None.  Since it's crucial that
    # _commit() finish successfully, we can't ignore shutdown races
    # here, and _commit() must not reference any globals.
    _os = _os       # for _commit()
    _io = _io       # for _commit()

    def __init__(self, filebasename, mode, flag='c'):
        filebasename = self._os.fsencode(filebasename)
        self._mode = mode
        self._readonly = (flag == 'r')

        # The directory file is a text file.  Each line looks like
        #    "%r, (%d, %d)\n" % (key, pos, siz)
        # where key is the string key, pos is the offset into the dat
        # file of the associated value's first byte, and siz is the number
        # of bytes in the associated value.
        self._dirfile = filebasename + b'.dir'

        # The data file is a binary file pointed into by the directory
        # file, and holds the values associated with keys.  Each value
        # begins at a _BLOCKSIZE-aligned byte offset, and is a raw
        # binary 8-bit string value.
        self._datfile = filebasename + b'.dat'
        self._bakfile = filebasename + b'.bak'

        # The index is an in-memory dict, mirroring the directory file.
        self._index = None  # maps keys to (pos, siz) pairs

        # Handle the creation
        self._create(flag)
        self._update(flag)

    def _create(self, flag):
        if flag == 'n':
            for filename in (self._datfile, self._bakfile, self._dirfile):
                try:
                    _os.remove(filename)
                except OSError:
                    pass
        # Mod by Jack: create data file if needed
        try:
            f = _io.open(self._datfile, 'r', encoding="Latin-1")
        except OSError:
            if flag not in ('c', 'n'):
                raise
            with _io.open(self._datfile, 'w', encoding="Latin-1") as f:
                self._chmod(self._datfile)
        else:
            f.close()

    # Read directory file into the in-memory index dict.
    def _update(self, flag):
        self._modified = False
        self._index = {}
        try:
            f = _io.open(self._dirfile, 'r', encoding="Latin-1")
        except OSError:
            if flag not in ('c', 'n'):
                raise
            self._modified = True
        else:
            with f:
                for line in f:
                    line = line.rstrip()
                    key, pos_and_siz_pair = _ast.literal_eval(line)
                    key = key.encode('Latin-1')
                    self._index[key] = pos_and_siz_pair

    # Write the index dict to the directory file.  The original directory
    # file (if any) is renamed with a .bak extension first.  If a .bak
    # file currently exists, it's deleted.
    def _commit(self):
        # CAUTION:  It's vital that _commit() succeed, and _commit() can
        # be called from __del__().  Therefore we must never reference a
        # global in this routine.
        if self._index is None or not self._modified:
            return  # nothing to do

        try:
            self._os.unlink(self._bakfile)
        except OSError:
            pass

        try:
            self._os.rename(self._dirfile, self._bakfile)
        except OSError:
            pass

        with self._io.open(self._dirfile, 'w', encoding="Latin-1") as f:
            self._chmod(self._dirfile)
            for key, pos_and_siz_pair in self._index.items():
                # Use Latin-1 since it has no qualms with any value in any
                # position; UTF-8, though, does care sometimes.
                entry = "%r, %r\n" % (key.decode('Latin-1'), pos_and_siz_pair)
                f.write(entry)

    sync = _commit

    def _verify_open(self):
        if self._index is None:
            raise error('DBM object has already been closed')

    def __getitem__(self, key):
        if isinstance(key, str):
            key = key.encode('utf-8')
        self._verify_open()
        pos, siz = self._index[key]     # may raise KeyError
        with _io.open(self._datfile, 'rb') as f:
            f.seek(pos)
            dat = f.read(siz)
        return dat

    # Append val to the data file, starting at a _BLOCKSIZE-aligned
    # offset.  The data file is first padded with NUL bytes (if needed)
    # to get to an aligned offset.  Return pair
    #     (starting offset of val, len(val))
    def _addval(self, val):
        with _io.open(self._datfile, 'rb+') as f:
            f.seek(0, 2)
            pos = int(f.tell())
            npos = ((pos + _BLOCKSIZE - 1) // _BLOCKSIZE) * _BLOCKSIZE
            f.write(b'\0'*(npos-pos))
            pos = npos
            f.write(val)
        return (pos, len(val))

    # Write val to the data file, starting at offset pos.  The caller
    # is responsible for ensuring that there's enough room starting at
    # pos to hold val, without overwriting some other value.  Return
    # pair (pos, len(val)).
    def _setval(self, pos, val):
        with _io.open(self._datfile, 'rb+') as f:
            f.seek(pos)
            f.write(val)
        return (pos, len(val))

    # key is a new key whose associated value starts in the data file
    # at offset pos and with length siz.  Add an index record to
    # the in-memory index dict, and append one to the directory file.
    def _addkey(self, key, pos_and_siz_pair):
        self._index[key] = pos_and_siz_pair
        with _io.open(self._dirfile, 'a', encoding="Latin-1") as f:
            self._chmod(self._dirfile)
            f.write("%r, %r\n" % (key.decode("Latin-1"), pos_and_siz_pair))

    def __setitem__(self, key, val):
        if self._readonly:
            raise error('The database is opened for reading only')
        if isinstance(key, str):
            key = key.encode('utf-8')
        elif not isinstance(key, (bytes, bytearray)):
            raise TypeError("keys must be bytes or strings")
        if isinstance(val, str):
            val = val.encode('utf-8')
        elif not isinstance(val, (bytes, bytearray)):
            raise TypeError("values must be bytes or strings")
        self._verify_open()
        self._modified = True
        if key not in self._index:
            self._addkey(key, self._addval(val))
        else:
            # See whether the new value is small enough to fit in the
            # (padded) space currently occupied by the old value.
            pos, siz = self._index[key]
            oldblocks = (siz + _BLOCKSIZE - 1) // _BLOCKSIZE
            newblocks = (len(val) + _BLOCKSIZE - 1) // _BLOCKSIZE
            if newblocks <= oldblocks:
                self._index[key] = self._setval(pos, val)
            else:
                # The new value doesn't fit in the (padded) space used
                # by the old value.  The blocks used by the old value are
                # forever lost.
                self._index[key] = self._addval(val)

            # Note that _index may be out of synch with the directory
            # file now:  _setval() and _addval() don't update the directory
            # file.  This also means that the on-disk directory and data
            # files are in a mutually inconsistent state, and they'll
            # remain that way until _commit() is called.  Note that this
            # is a disaster (for the database) if the program crashes
            # (so that _commit() never gets called).

    def __delitem__(self, key):
        if self._readonly:
            raise error('The database is opened for reading only')
        if isinstance(key, str):
            key = key.encode('utf-8')
        self._verify_open()
        self._modified = True
        # The blocks used by the associated value are lost.
        del self._index[key]
        # XXX It's unclear why we do a _commit() here (the code always
        # XXX has, so I'm not changing it).  __setitem__ doesn't try to
        # XXX keep the directory file in synch.  Why should we?  Or
        # XXX why shouldn't __setitem__?
        self._commit()

    def keys(self):
        try:
            return list(self._index)
        except TypeError:
            raise error('DBM object has already been closed') from None

    def items(self):
        self._verify_open()
        return [(key, self[key]) for key in self._index.keys()]

    def __contains__(self, key):
        if isinstance(key, str):
            key = key.encode('utf-8')
        try:
            return key in self._index
        except TypeError:
            if self._index is None:
                raise error('DBM object has already been closed') from None
            else:
                raise

    def iterkeys(self):
        try:
            return iter(self._index)
        except TypeError:
            raise error('DBM object has already been closed') from None
    __iter__ = iterkeys

    def __len__(self):
        try:
            return len(self._index)
        except TypeError:
            raise error('DBM object has already been closed') from None

    def close(self):
        try:
            self._commit()
        finally:
            self._index = self._datfile = self._dirfile = self._bakfile = None

    __del__ = close

    def _chmod(self, file):
        self._os.chmod(file, self._mode)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def open(file, flag='c', mode=0o666):
    """Open the database file, filename, and return corresponding object.

    The flag argument, used to control how the database is opened in the
    other DBM implementations, supports only the semantics of 'c' and 'n'
    values.  Other values will default to the semantics of 'c' value:
    the database will always opened for update and will be created if it
    does not exist.

    The optional mode argument is the UNIX mode of the file, used only when
    the database has to be created.  It defaults to octal code 0o666 (and
    will be modified by the prevailing umask).

    """

    # Modify mode depending on the umask
    try:
        um = _os.umask(0)
        _os.umask(um)
    except AttributeError:
        pass
    else:
        # Turn off any bits that are set in the umask
        mode = mode & (~um)
    if flag not in ('r', 'w', 'c', 'n'):
        raise ValueError("Flag must be one of 'r', 'w', 'c', or 'n'")
    return _Database(file, mode, flag=flag)
