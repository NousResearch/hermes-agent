"""Routines to help recognizing sound files.

Function whathdr() recognizes various types of sound file headers.
It understands almost all headers that SOX can decode.

The return tuple contains the following items, in this order:
- file type (as SOX understands it)
- sampling rate (0 if unknown or hard to decode)
- number of channels (0 if unknown or hard to decode)
- number of frames in the file (-1 if unknown or hard to decode)
- number of bits/sample, or 'U' for U-LAW, or 'A' for A-LAW

If the file doesn't have a recognizable type, it returns None.
If the file can't be opened, OSError is raised.

To compute the total time, divide the number of frames by the
sampling rate (a frame contains a sample for each channel).

Function what() calls whathdr().  (It used to also use some
heuristics for raw data, but this doesn't work very well.)

Finally, the function test() is a simple main program that calls
what() for all files mentioned on the argument list.  For directory
arguments it calls what() for all files in that directory.  Default
argument is "." (testing all files in the current directory).  The
option -r tells it to recurse down directories found inside
explicitly given directories.
"""

import warnings

warnings._deprecated(__name__, remove=(3, 13))

# The file structure is top-down except that the test program and its
# subroutine come last.

__all__ = ['what', 'whathdr']

from collections import namedtuple

SndHeaders = namedtuple('SndHeaders',
                        'filetype framerate nchannels nframes sampwidth')

SndHeaders.filetype.__doc__ = ("""The value for type indicates the data type
and will be one of the strings 'aifc', 'aiff', 'au','hcom',
'sndr', 'sndt', 'voc', 'wav', '8svx', 'sb', 'ub', or 'ul'.""")
SndHeaders.framerate.__doc__ = ("""The sampling_rate will be either the actual
value or 0 if unknown or difficult to decode.""")
SndHeaders.nchannels.__doc__ = ("""The number of channels or 0 if it cannot be
determined or if the value is difficult to decode.""")
SndHeaders.nframes.__doc__ = ("""The value for frames will be either the number
of frames or -1.""")
SndHeaders.sampwidth.__doc__ = ("""Either the sample size in bits or
'A' for A-LAW or 'U' for u-LAW.""")

def what(filename):
    """Guess the type of a sound file."""
    res = whathdr(filename)
    return res


def whathdr(filename):
    """Recognize sound headers."""
    with open(filename, 'rb') as f:
        h = f.read(512)
        for tf in tests:
            res = tf(h, f)
            if res:
                return SndHeaders(*res)
        return None


#-----------------------------------#
# Subroutines per sound header type #
#-----------------------------------#

tests = []

def test_aifc(h, f):
    """AIFC and AIFF files"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        import aifc
    if not h.startswith(b'FORM'):
        return None
    if h[8:12] == b'AIFC':
        fmt = 'aifc'
    elif h[8:12] == b'AIFF':
        fmt = 'aiff'
    else:
        return None
    f.seek(0)
    try:
        a = aifc.open(f, 'r')
    except (EOFError, aifc.Error):
        return None
    return (fmt, a.getframerate(), a.getnchannels(),
            a.getnframes(), 8 * a.getsampwidth())

tests.append(test_aifc)


def test_au(h, f):
    """AU and SND files"""
    if h.startswith(b'.snd'):
        func = get_long_be
    elif h[:4] in (b'\0ds.', b'dns.'):
        func = get_long_le
    else:
        return None
    filetype = 'au'
    hdr_size = func(h[4:8])
    data_size = func(h[8:12])
    encoding = func(h[12:16])
    rate = func(h[16:20])
    nchannels = func(h[20:24])
    sample_size = 1 # default
    if encoding == 1:
        sample_bits = 'U'
    elif encoding == 2:
        sample_bits = 8
    elif encoding == 3:
        sample_bits = 16
        sample_size = 2
    else:
        sample_bits = '?'
    frame_size = sample_size * nchannels
    if frame_size:
        nframe = data_size / frame_size
    else:
        nframe = -1
    return filetype, rate, nchannels, nframe, sample_bits

tests.append(test_au)


def test_hcom(h, f):
    """HCOM file"""
    if h[65:69] != b'FSSD' or h[128:132] != b'HCOM':
        return None
    divisor = get_long_be(h[144:148])
    if divisor:
        rate = 22050 / divisor
    else:
        rate = 0
    return 'hcom', rate, 1, -1, 8

tests.append(test_hcom)


def test_voc(h, f):
    """VOC file"""
    if not h.startswith(b'Creative Voice File\032'):
        return None
    sbseek = get_short_le(h[20:22])
    rate = 0
    if 0 <= sbseek < 500 and h[sbseek] == 1:
        ratecode = 256 - h[sbseek+4]
        if ratecode:
            rate = int(1000000.0 / ratecode)
    return 'voc', rate, 1, -1, 8

tests.append(test_voc)


def test_wav(h, f):
    """WAV file"""
    import wave
    # 'RIFF' <len> 'WAVE' 'fmt ' <len>
    if not h.startswith(b'RIFF') or h[8:12] != b'WAVE' or h[12:16] != b'fmt ':
        return None
    f.seek(0)
    try:
        w = wave.open(f, 'r')
    except (EOFError, wave.Error):
        return None
    return ('wav', w.getframerate(), w.getnchannels(),
                   w.getnframes(), 8*w.getsampwidth())

tests.append(test_wav)


def test_8svx(h, f):
    """8SVX file"""
    if not h.startswith(b'FORM') or h[8:12] != b'8SVX':
        return None
    # Should decode it to get #channels -- assume always 1
    return '8svx', 0, 1, 0, 8

tests.append(test_8svx)


def test_sndt(h, f):
    """SNDT file"""
    if h.startswith(b'SOUND'):
        nsamples = get_long_le(h[8:12])
        rate = get_short_le(h[20:22])
        return 'sndt', rate, 1, nsamples, 8

tests.append(test_sndt)


def test_sndr(h, f):
    """SNDR file"""
    if h.startswith(b'\0\0'):
        rate = get_short_le(h[2:4])
        if 4000 <= rate <= 25000:
            return 'sndr', rate, 1, -1, 8

tests.append(test_sndr)


#-------------------------------------------#
# Subroutines to extract numbers from bytes #
#-------------------------------------------#

def get_long_be(b):
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]

def get_long_le(b):
    return (b[3] << 24) | (b[2] << 16) | (b[1] << 8) | b[0]

def get_short_be(b):
    return (b[0] << 8) | b[1]

def get_short_le(b):
    return (b[1] << 8) | b[0]


#--------------------#
# Small test program #
#--------------------#

def test():
    import sys
    recursive = 0
    if sys.argv[1:] and sys.argv[1] == '-r':
        del sys.argv[1:2]
        recursive = 1
    try:
        if sys.argv[1:]:
            testall(sys.argv[1:], recursive, 1)
        else:
            testall(['.'], recursive, 1)
    except KeyboardInterrupt:
        sys.stderr.write('\n[Interrupted]\n')
        sys.exit(1)

def testall(list, recursive, toplevel):
    import sys
    import os
    for filename in list:
        if os.path.isdir(filename):
            print(filename + '/:', end=' ')
            if recursive or toplevel:
                print('recursing down:')
                import glob
                names = glob.glob(os.path.join(glob.escape(filename), '*'))
                testall(names, recursive, 0)
            else:
                print('*** directory (use -r) ***')
        else:
            print(filename + ':', end=' ')
            sys.stdout.flush()
            try:
                print(what(filename))
            except OSError:
                print('*** not found ***')

if __name__ == '__main__':
    test()
