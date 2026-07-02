# This module implements the RFCs 3490 (IDNA) and 3491 (Nameprep)

import stringprep, re, codecs
from unicodedata import ucd_3_2_0 as unicodedata

# IDNA section 3.1
dots = re.compile("[\u002E\u3002\uFF0E\uFF61]")

# IDNA section 5
ace_prefix = b"xn--"
sace_prefix = "xn--"

# This assumes query strings, so AllowUnassigned is true
def nameprep(label):
    # Map
    newlabel = []
    for c in label:
        if stringprep.in_table_b1(c):
            # Map to nothing
            continue
        newlabel.append(stringprep.map_table_b2(c))
    label = "".join(newlabel)

    # Normalize
    label = unicodedata.normalize("NFKC", label)

    # Prohibit
    for c in label:
        if stringprep.in_table_c12(c) or \
           stringprep.in_table_c22(c) or \
           stringprep.in_table_c3(c) or \
           stringprep.in_table_c4(c) or \
           stringprep.in_table_c5(c) or \
           stringprep.in_table_c6(c) or \
           stringprep.in_table_c7(c) or \
           stringprep.in_table_c8(c) or \
           stringprep.in_table_c9(c):
            raise UnicodeError("Invalid character %r" % c)

    # Check bidi
    RandAL = [stringprep.in_table_d1(x) for x in label]
    if any(RandAL):
        # There is a RandAL char in the string. Must perform further
        # tests:
        # 1) The characters in section 5.8 MUST be prohibited.
        # This is table C.8, which was already checked
        # 2) If a string contains any RandALCat character, the string
        # MUST NOT contain any LCat character.
        if any(stringprep.in_table_d2(x) for x in label):
            raise UnicodeError("Violation of BIDI requirement 2")
        # 3) If a string contains any RandALCat character, a
        # RandALCat character MUST be the first character of the
        # string, and a RandALCat character MUST be the last
        # character of the string.
        if not RandAL[0] or not RandAL[-1]:
            raise UnicodeError("Violation of BIDI requirement 3")

    return label

def ToASCII(label):
    try:
        # Step 1: try ASCII
        label = label.encode("ascii")
    except UnicodeError:
        pass
    else:
        # Skip to step 3: UseSTD3ASCIIRules is false, so
        # Skip to step 8.
        if 0 < len(label) < 64:
            return label
        raise UnicodeError("label empty or too long")

    # Step 2: nameprep
    label = nameprep(label)

    # Step 3: UseSTD3ASCIIRules is false
    # Step 4: try ASCII
    try:
        label = label.encode("ascii")
    except UnicodeError:
        pass
    else:
        # Skip to step 8.
        if 0 < len(label) < 64:
            return label
        raise UnicodeError("label empty or too long")

    # Step 5: Check ACE prefix
    if label.startswith(sace_prefix):
        raise UnicodeError("Label starts with ACE prefix")

    # Step 6: Encode with PUNYCODE
    label = label.encode("punycode")

    # Step 7: Prepend ACE prefix
    label = ace_prefix + label

    # Step 8: Check size
    if 0 < len(label) < 64:
        return label
    raise UnicodeError("label empty or too long")

def ToUnicode(label):
    # Step 1: Check for ASCII
    if isinstance(label, bytes):
        pure_ascii = True
    else:
        try:
            label = label.encode("ascii")
            pure_ascii = True
        except UnicodeError:
            pure_ascii = False
    if not pure_ascii:
        # Step 2: Perform nameprep
        label = nameprep(label)
        # It doesn't say this, but apparently, it should be ASCII now
        try:
            label = label.encode("ascii")
        except UnicodeError:
            raise UnicodeError("Invalid character in IDN label")
    # Step 3: Check for ACE prefix
    if not label.startswith(ace_prefix):
        return str(label, "ascii")

    # Step 4: Remove ACE prefix
    label1 = label[len(ace_prefix):]

    # Step 5: Decode using PUNYCODE
    result = label1.decode("punycode")

    # Step 6: Apply ToASCII
    label2 = ToASCII(result)

    # Step 7: Compare the result of step 6 with the one of step 3
    # label2 will already be in lower case.
    if str(label, "ascii").lower() != str(label2, "ascii"):
        raise UnicodeError("IDNA does not round-trip", label, label2)

    # Step 8: return the result of step 5
    return result

### Codec APIs

class Codec(codecs.Codec):
    def encode(self, input, errors='strict'):

        if errors != 'strict':
            # IDNA is quite clear that implementations must be strict
            raise UnicodeError("unsupported error handling "+errors)

        if not input:
            return b'', 0

        try:
            result = input.encode('ascii')
        except UnicodeEncodeError:
            pass
        else:
            # ASCII name: fast path
            labels = result.split(b'.')
            for label in labels[:-1]:
                if not (0 < len(label) < 64):
                    raise UnicodeError("label empty or too long")
            if len(labels[-1]) >= 64:
                raise UnicodeError("label too long")
            return result, len(input)

        result = bytearray()
        labels = dots.split(input)
        if labels and not labels[-1]:
            trailing_dot = b'.'
            del labels[-1]
        else:
            trailing_dot = b''
        for label in labels:
            if result:
                # Join with U+002E
                result.extend(b'.')
            result.extend(ToASCII(label))
        return bytes(result+trailing_dot), len(input)

    def decode(self, input, errors='strict'):

        if errors != 'strict':
            raise UnicodeError("Unsupported error handling "+errors)

        if not input:
            return "", 0

        # IDNA allows decoding to operate on Unicode strings, too.
        if not isinstance(input, bytes):
            # XXX obviously wrong, see #3232
            input = bytes(input)

        if ace_prefix not in input:
            # Fast path
            try:
                return input.decode('ascii'), len(input)
            except UnicodeDecodeError:
                pass

        labels = input.split(b".")

        if labels and len(labels[-1]) == 0:
            trailing_dot = '.'
            del labels[-1]
        else:
            trailing_dot = ''

        result = []
        for label in labels:
            result.append(ToUnicode(label))

        return ".".join(result)+trailing_dot, len(input)

class IncrementalEncoder(codecs.BufferedIncrementalEncoder):
    def _buffer_encode(self, input, errors, final):
        if errors != 'strict':
            # IDNA is quite clear that implementations must be strict
            raise UnicodeError("unsupported error handling "+errors)

        if not input:
            return (b'', 0)

        labels = dots.split(input)
        trailing_dot = b''
        if labels:
            if not labels[-1]:
                trailing_dot = b'.'
                del labels[-1]
            elif not final:
                # Keep potentially unfinished label until the next call
                del labels[-1]
                if labels:
                    trailing_dot = b'.'

        result = bytearray()
        size = 0
        for label in labels:
            if size:
                # Join with U+002E
                result.extend(b'.')
                size += 1
            result.extend(ToASCII(label))
            size += len(label)

        result += trailing_dot
        size += len(trailing_dot)
        return (bytes(result), size)

class IncrementalDecoder(codecs.BufferedIncrementalDecoder):
    def _buffer_decode(self, input, errors, final):
        if errors != 'strict':
            raise UnicodeError("Unsupported error handling "+errors)

        if not input:
            return ("", 0)

        # IDNA allows decoding to operate on Unicode strings, too.
        if isinstance(input, str):
            labels = dots.split(input)
        else:
            # Must be ASCII string
            input = str(input, "ascii")
            labels = input.split(".")

        trailing_dot = ''
        if labels:
            if not labels[-1]:
                trailing_dot = '.'
                del labels[-1]
            elif not final:
                # Keep potentially unfinished label until the next call
                del labels[-1]
                if labels:
                    trailing_dot = '.'

        result = []
        size = 0
        for label in labels:
            result.append(ToUnicode(label))
            if size:
                size += 1
            size += len(label)

        result = ".".join(result) + trailing_dot
        size += len(trailing_dot)
        return (result, size)

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='idna',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamwriter=StreamWriter,
        streamreader=StreamReader,
    )
