"""Stuff to parse Sun and NeXT audio files.

An audio file consists of a header followed by the data.  The structure
of the header is as follows.

        +---------------+
        | magic word    |
        +---------------+
        | header size   |
        +---------------+
        | data size     |
        +---------------+
        | encoding      |
        +---------------+
        | sample rate   |
        +---------------+
        | # of channels |
        +---------------+
        | info          |
        |               |
        +---------------+

The magic word consists of the 4 characters '.snd'.  Apart from the
info field, all header fields are 4 bytes in size.  They are all
32-bit unsigned integers encoded in big-endian byte order.

The header size really gives the start of the data.
The data size is the physical size of the data.  From the other
parameters the number of frames can be calculated.
The encoding gives the way in which audio samples are encoded.
Possible values are listed below.
The info field currently consists of an ASCII string giving a
human-readable description of the audio file.  The info field is
padded with NUL bytes to the header size.

Usage.

Reading audio files:
        f = sunau.open(file, 'r')
where file is either the name of a file or an open file pointer.
The open file pointer must have methods read(), seek(), and close().
When the setpos() and rewind() methods are not used, the seek()
method is not  necessary.

This returns an instance of a class with the following public methods:
        getnchannels()  -- returns number of audio channels (1 for
                           mono, 2 for stereo)
        getsampwidth()  -- returns sample width in bytes
        getframerate()  -- returns sampling frequency
        getnframes()    -- returns number of audio frames
        getcomptype()   -- returns compression type ('NONE' or 'ULAW')
        getcompname()   -- returns human-readable version of
                           compression type ('not compressed' matches 'NONE')
        getparams()     -- returns a namedtuple consisting of all of the
                           above in the above order
        getmarkers()    -- returns None (for compatibility with the
                           aifc module)
        getmark(id)     -- raises an error since the mark does not
                           exist (for compatibility with the aifc module)
        readframes(n)   -- returns at most n frames of audio
        rewind()        -- rewind to the beginning of the audio stream
        setpos(pos)     -- seek to the specified position
        tell()          -- return the current position
        close()         -- close the instance (make it unusable)
The position returned by tell() and the position given to setpos()
are compatible and have nothing to do with the actual position in the
file.
The close() method is called automatically when the class instance
is destroyed.

Writing audio files:
        f = sunau.open(file, 'w')
where file is either the name of a file or an open file pointer.
The open file pointer must have methods write(), tell(), seek(), and
close().

This returns an instance of a class with the following public methods:
        setnchannels(n) -- set the number of channels
        setsampwidth(n) -- set the sample width
        setframerate(n) -- set the frame rate
        setnframes(n)   -- set the number of frames
        setcomptype(type, name)
                        -- set the compression type and the
                           human-readable compression type
        setparams(tuple)-- set all parameters at once
        tell()          -- return current position in output file
        writeframesraw(data)
                        -- write audio frames without pathing up the
                           file header
        writeframes(data)
                        -- write audio frames and patch up the file header
        close()         -- patch up the file header and close the
                           output file
You should set the parameters before the first writeframesraw or
writeframes.  The total number of frames does not need to be set,
but when it is set to the correct value, the header does not have to
be patched up.
It is best to first set all parameters, perhaps possibly the
compression type, and then write audio frames using writeframesraw.
When all frames have been written, either call writeframes(b'') or
close() to patch up the sizes in the header.
The close() method is called automatically when the class instance
is destroyed.
"""

from collections import namedtuple
import warnings

warnings._deprecated(__name__, remove=(3, 13))


_sunau_params = namedtuple('_sunau_params',
                           'nchannels sampwidth framerate nframes comptype compname')

# from <multimedia/audio_filehdr.h>
AUDIO_FILE_MAGIC = 0x2e736e64
AUDIO_FILE_ENCODING_MULAW_8 = 1
AUDIO_FILE_ENCODING_LINEAR_8 = 2
AUDIO_FILE_ENCODING_LINEAR_16 = 3
AUDIO_FILE_ENCODING_LINEAR_24 = 4
AUDIO_FILE_ENCODING_LINEAR_32 = 5
AUDIO_FILE_ENCODING_FLOAT = 6
AUDIO_FILE_ENCODING_DOUBLE = 7
AUDIO_FILE_ENCODING_ADPCM_G721 = 23
AUDIO_FILE_ENCODING_ADPCM_G722 = 24
AUDIO_FILE_ENCODING_ADPCM_G723_3 = 25
AUDIO_FILE_ENCODING_ADPCM_G723_5 = 26
AUDIO_FILE_ENCODING_ALAW_8 = 27

# from <multimedia/audio_hdr.h>
AUDIO_UNKNOWN_SIZE = 0xFFFFFFFF        # ((unsigned)(~0))

_simple_encodings = [AUDIO_FILE_ENCODING_MULAW_8,
                     AUDIO_FILE_ENCODING_LINEAR_8,
                     AUDIO_FILE_ENCODING_LINEAR_16,
                     AUDIO_FILE_ENCODING_LINEAR_24,
                     AUDIO_FILE_ENCODING_LINEAR_32,
                     AUDIO_FILE_ENCODING_ALAW_8]

class Error(Exception):
    pass

def _read_u32(file):
    x = 0
    for i in range(4):
        byte = file.read(1)
        if not byte:
            raise EOFError
        x = x*256 + ord(byte)
    return x

def _write_u32(file, x):
    data = []
    for i in range(4):
        d, m = divmod(x, 256)
        data.insert(0, int(m))
        x = d
    file.write(bytes(data))

class Au_read:

    def __init__(self, f):
        if type(f) == type(''):
            import builtins
            f = builtins.open(f, 'rb')
            self._opened = True
        else:
            self._opened = False
        self.initfp(f)

    def __del__(self):
        if self._file:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def initfp(self, file):
        self._file = file
        self._soundpos = 0
        magic = int(_read_u32(file))
        if magic != AUDIO_FILE_MAGIC:
            raise Error('bad magic number')
        self._hdr_size = int(_read_u32(file))
        if self._hdr_size < 24:
            raise Error('header size too small')
        if self._hdr_size > 100:
            raise Error('header size ridiculously large')
        self._data_size = _read_u32(file)
        if self._data_size != AUDIO_UNKNOWN_SIZE:
            self._data_size = int(self._data_size)
        self._encoding = int(_read_u32(file))
        if self._encoding not in _simple_encodings:
            raise Error('encoding not (yet) supported')
        if self._encoding in (AUDIO_FILE_ENCODING_MULAW_8,
                  AUDIO_FILE_ENCODING_ALAW_8):
            self._sampwidth = 2
            self._framesize = 1
        elif self._encoding == AUDIO_FILE_ENCODING_LINEAR_8:
            self._framesize = self._sampwidth = 1
        elif self._encoding == AUDIO_FILE_ENCODING_LINEAR_16:
            self._framesize = self._sampwidth = 2
        elif self._encoding == AUDIO_FILE_ENCODING_LINEAR_24:
            self._framesize = self._sampwidth = 3
        elif self._encoding == AUDIO_FILE_ENCODING_LINEAR_32:
            self._framesize = self._sampwidth = 4
        else:
            raise Error('unknown encoding')
        self._framerate = int(_read_u32(file))
        self._nchannels = int(_read_u32(file))
        if not self._nchannels:
            raise Error('bad # of channels')
        self._framesize = self._framesize * self._nchannels
        if self._hdr_size > 24:
            self._info = file.read(self._hdr_size - 24)
            self._info, _, _ = self._info.partition(b'\0')
        else:
            self._info = b''
        try:
            self._data_pos = file.tell()
        except (AttributeError, OSError):
            self._data_pos = None

    def getfp(self):
        return self._file

    def getnchannels(self):
        return self._nchannels

    def getsampwidth(self):
        return self._sampwidth

    def getframerate(self):
        return self._framerate

    def getnframes(self):
        if self._data_size == AUDIO_UNKNOWN_SIZE:
            return AUDIO_UNKNOWN_SIZE
        if self._encoding in _simple_encodings:
            return self._data_size // self._framesize
        return 0                # XXX--must do some arithmetic here

    def getcomptype(self):
        if self._encoding == AUDIO_FILE_ENCODING_MULAW_8:
            return 'ULAW'
        elif self._encoding == AUDIO_FILE_ENCODING_ALAW_8:
            return 'ALAW'
        else:
            return 'NONE'

    def getcompname(self):
        if self._encoding == AUDIO_FILE_ENCODING_MULAW_8:
            return 'CCITT G.711 u-law'
        elif self._encoding == AUDIO_FILE_ENCODING_ALAW_8:
            return 'CCITT G.711 A-law'
        else:
            return 'not compressed'

    def getparams(self):
        return _sunau_params(self.getnchannels(), self.getsampwidth(),
                  self.getframerate(), self.getnframes(),
                  self.getcomptype(), self.getcompname())

    def getmarkers(self):
        return None

    def getmark(self, id):
        raise Error('no marks')

    def readframes(self, nframes):
        if self._encoding in _simple_encodings:
            if nframes == AUDIO_UNKNOWN_SIZE:
                data = self._file.read()
            else:
                data = self._file.read(nframes * self._framesize)
            self._soundpos += len(data) // self._framesize
            if self._encoding == AUDIO_FILE_ENCODING_MULAW_8:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=DeprecationWarning)
                    import audioop
                data = audioop.ulaw2lin(data, self._sampwidth)
            return data
        return None             # XXX--not implemented yet

    def rewind(self):
        if self._data_pos is None:
            raise OSError('cannot seek')
        self._file.seek(self._data_pos)
        self._soundpos = 0

    def tell(self):
        return self._soundpos

    def setpos(self, pos):
        if pos < 0 or pos > self.getnframes():
            raise Error('position not in range')
        if self._data_pos is None:
            raise OSError('cannot seek')
        self._file.seek(self._data_pos + pos * self._framesize)
        self._soundpos = pos

    def close(self):
        file = self._file
        if file:
            self._file = None
            if self._opened:
                file.close()

class Au_write:

    def __init__(self, f):
        if type(f) == type(''):
            import builtins
            f = builtins.open(f, 'wb')
            self._opened = True
        else:
            self._opened = False
        self.initfp(f)

    def __del__(self):
        if self._file:
            self.close()
        self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def initfp(self, file):
        self._file = file
        self._framerate = 0
        self._nchannels = 0
        self._sampwidth = 0
        self._framesize = 0
        self._nframes = AUDIO_UNKNOWN_SIZE
        self._nframeswritten = 0
        self._datawritten = 0
        self._datalength = 0
        self._info = b''
        self._comptype = 'ULAW' # default is U-law

    def setnchannels(self, nchannels):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if nchannels not in (1, 2, 4):
            raise Error('only 1, 2, or 4 channels supported')
        self._nchannels = nchannels

    def getnchannels(self):
        if not self._nchannels:
            raise Error('number of channels not set')
        return self._nchannels

    def setsampwidth(self, sampwidth):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if sampwidth not in (1, 2, 3, 4):
            raise Error('bad sample width')
        self._sampwidth = sampwidth

    def getsampwidth(self):
        if not self._framerate:
            raise Error('sample width not specified')
        return self._sampwidth

    def setframerate(self, framerate):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        self._framerate = framerate

    def getframerate(self):
        if not self._framerate:
            raise Error('frame rate not set')
        return self._framerate

    def setnframes(self, nframes):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if nframes < 0:
            raise Error('# of frames cannot be negative')
        self._nframes = nframes

    def getnframes(self):
        return self._nframeswritten

    def setcomptype(self, type, name):
        if type in ('NONE', 'ULAW'):
            self._comptype = type
        else:
            raise Error('unknown compression type')

    def getcomptype(self):
        return self._comptype

    def getcompname(self):
        if self._comptype == 'ULAW':
            return 'CCITT G.711 u-law'
        elif self._comptype == 'ALAW':
            return 'CCITT G.711 A-law'
        else:
            return 'not compressed'

    def setparams(self, params):
        nchannels, sampwidth, framerate, nframes, comptype, compname = params
        self.setnchannels(nchannels)
        self.setsampwidth(sampwidth)
        self.setframerate(framerate)
        self.setnframes(nframes)
        self.setcomptype(comptype, compname)

    def getparams(self):
        return _sunau_params(self.getnchannels(), self.getsampwidth(),
                  self.getframerate(), self.getnframes(),
                  self.getcomptype(), self.getcompname())

    def tell(self):
        return self._nframeswritten

    def writeframesraw(self, data):
        if not isinstance(data, (bytes, bytearray)):
            data = memoryview(data).cast('B')
        self._ensure_header_written()
        if self._comptype == 'ULAW':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=DeprecationWarning)
                import audioop
            data = audioop.lin2ulaw(data, self._sampwidth)
        nframes = len(data) // self._framesize
        self._file.write(data)
        self._nframeswritten = self._nframeswritten + nframes
        self._datawritten = self._datawritten + len(data)

    def writeframes(self, data):
        self.writeframesraw(data)
        if self._nframeswritten != self._nframes or \
                  self._datalength != self._datawritten:
            self._patchheader()

    def close(self):
        if self._file:
            try:
                self._ensure_header_written()
                if self._nframeswritten != self._nframes or \
                        self._datalength != self._datawritten:
                    self._patchheader()
                self._file.flush()
            finally:
                file = self._file
                self._file = None
                if self._opened:
                    file.close()

    #
    # private methods
    #

    def _ensure_header_written(self):
        if not self._nframeswritten:
            if not self._nchannels:
                raise Error('# of channels not specified')
            if not self._sampwidth:
                raise Error('sample width not specified')
            if not self._framerate:
                raise Error('frame rate not specified')
            self._write_header()

    def _write_header(self):
        if self._comptype == 'NONE':
            if self._sampwidth == 1:
                encoding = AUDIO_FILE_ENCODING_LINEAR_8
                self._framesize = 1
            elif self._sampwidth == 2:
                encoding = AUDIO_FILE_ENCODING_LINEAR_16
                self._framesize = 2
            elif self._sampwidth == 3:
                encoding = AUDIO_FILE_ENCODING_LINEAR_24
                self._framesize = 3
            elif self._sampwidth == 4:
                encoding = AUDIO_FILE_ENCODING_LINEAR_32
                self._framesize = 4
            else:
                raise Error('internal error')
        elif self._comptype == 'ULAW':
            encoding = AUDIO_FILE_ENCODING_MULAW_8
            self._framesize = 1
        else:
            raise Error('internal error')
        self._framesize = self._framesize * self._nchannels
        _write_u32(self._file, AUDIO_FILE_MAGIC)
        header_size = 25 + len(self._info)
        header_size = (header_size + 7) & ~7
        _write_u32(self._file, header_size)
        if self._nframes == AUDIO_UNKNOWN_SIZE:
            length = AUDIO_UNKNOWN_SIZE
        else:
            length = self._nframes * self._framesize
        try:
            self._form_length_pos = self._file.tell()
        except (AttributeError, OSError):
            self._form_length_pos = None
        _write_u32(self._file, length)
        self._datalength = length
        _write_u32(self._file, encoding)
        _write_u32(self._file, self._framerate)
        _write_u32(self._file, self._nchannels)
        self._file.write(self._info)
        self._file.write(b'\0'*(header_size - len(self._info) - 24))

    def _patchheader(self):
        if self._form_length_pos is None:
            raise OSError('cannot seek')
        self._file.seek(self._form_length_pos)
        _write_u32(self._file, self._datawritten)
        self._datalength = self._datawritten
        self._file.seek(0, 2)

def open(f, mode=None):
    if mode is None:
        if hasattr(f, 'mode'):
            mode = f.mode
        else:
            mode = 'rb'
    if mode in ('r', 'rb'):
        return Au_read(f)
    elif mode in ('w', 'wb'):
        return Au_write(f)
    else:
        raise Error("mode must be 'r', 'rb', 'w', or 'wb'")
