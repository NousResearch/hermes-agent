"""Python 'bz2_codec' Codec - bz2 compression encoding.

This codec de/encodes from bytes to bytes and is therefore usable with
bytes.transform() and bytes.untransform().

Adapted by Raymond Hettinger from zlib_codec.py which was written
by Marc-Andre Lemburg (mal@lemburg.com).
"""

import codecs
import bz2 # this codec needs the optional bz2 module !

### Codec APIs

def bz2_encode(input, errors='strict'):
    assert errors == 'strict'
    return (bz2.compress(input), len(input))

def bz2_decode(input, errors='strict'):
    assert errors == 'strict'
    return (bz2.decompress(input), len(input))

class Codec(codecs.Codec):
    def encode(self, input, errors='strict'):
        return bz2_encode(input, errors)
    def decode(self, input, errors='strict'):
        return bz2_decode(input, errors)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def __init__(self, errors='strict'):
        assert errors == 'strict'
        self.errors = errors
        self.compressobj = bz2.BZ2Compressor()

    def encode(self, input, final=False):
        if final:
            c = self.compressobj.compress(input)
            return c + self.compressobj.flush()
        else:
            return self.compressobj.compress(input)

    def reset(self):
        self.compressobj = bz2.BZ2Compressor()

class IncrementalDecoder(codecs.IncrementalDecoder):
    def __init__(self, errors='strict'):
        assert errors == 'strict'
        self.errors = errors
        self.decompressobj = bz2.BZ2Decompressor()

    def decode(self, input, final=False):
        try:
            return self.decompressobj.decompress(input)
        except EOFError:
            return ''

    def reset(self):
        self.decompressobj = bz2.BZ2Decompressor()

class StreamWriter(Codec, codecs.StreamWriter):
    charbuffertype = bytes

class StreamReader(Codec, codecs.StreamReader):
    charbuffertype = bytes

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name="bz2",
        encode=bz2_encode,
        decode=bz2_decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamwriter=StreamWriter,
        streamreader=StreamReader,
        _is_text_encoding=False,
    )
