""" Python Character Mapping Codec cp875 generated from 'MAPPINGS/VENDORS/MICSFT/EBCDIC/CP875.TXT' with gencodec.py.

"""#"

import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return codecs.charmap_encode(input,errors,encoding_table)

    def decode(self,input,errors='strict'):
        return codecs.charmap_decode(input,errors,decoding_table)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.charmap_encode(input,self.errors,encoding_table)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.charmap_decode(input,self.errors,decoding_table)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='cp875',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    '\x00'     #  0x00 -> NULL
    '\x01'     #  0x01 -> START OF HEADING
    '\x02'     #  0x02 -> START OF TEXT
    '\x03'     #  0x03 -> END OF TEXT
    '\x9c'     #  0x04 -> CONTROL
    '\t'       #  0x05 -> HORIZONTAL TABULATION
    '\x86'     #  0x06 -> CONTROL
    '\x7f'     #  0x07 -> DELETE
    '\x97'     #  0x08 -> CONTROL
    '\x8d'     #  0x09 -> CONTROL
    '\x8e'     #  0x0A -> CONTROL
    '\x0b'     #  0x0B -> VERTICAL TABULATION
    '\x0c'     #  0x0C -> FORM FEED
    '\r'       #  0x0D -> CARRIAGE RETURN
    '\x0e'     #  0x0E -> SHIFT OUT
    '\x0f'     #  0x0F -> SHIFT IN
    '\x10'     #  0x10 -> DATA LINK ESCAPE
    '\x11'     #  0x11 -> DEVICE CONTROL ONE
    '\x12'     #  0x12 -> DEVICE CONTROL TWO
    '\x13'     #  0x13 -> DEVICE CONTROL THREE
    '\x9d'     #  0x14 -> CONTROL
    '\x85'     #  0x15 -> CONTROL
    '\x08'     #  0x16 -> BACKSPACE
    '\x87'     #  0x17 -> CONTROL
    '\x18'     #  0x18 -> CANCEL
    '\x19'     #  0x19 -> END OF MEDIUM
    '\x92'     #  0x1A -> CONTROL
    '\x8f'     #  0x1B -> CONTROL
    '\x1c'     #  0x1C -> FILE SEPARATOR
    '\x1d'     #  0x1D -> GROUP SEPARATOR
    '\x1e'     #  0x1E -> RECORD SEPARATOR
    '\x1f'     #  0x1F -> UNIT SEPARATOR
    '\x80'     #  0x20 -> CONTROL
    '\x81'     #  0x21 -> CONTROL
    '\x82'     #  0x22 -> CONTROL
    '\x83'     #  0x23 -> CONTROL
    '\x84'     #  0x24 -> CONTROL
    '\n'       #  0x25 -> LINE FEED
    '\x17'     #  0x26 -> END OF TRANSMISSION BLOCK
    '\x1b'     #  0x27 -> ESCAPE
    '\x88'     #  0x28 -> CONTROL
    '\x89'     #  0x29 -> CONTROL
    '\x8a'     #  0x2A -> CONTROL
    '\x8b'     #  0x2B -> CONTROL
    '\x8c'     #  0x2C -> CONTROL
    '\x05'     #  0x2D -> ENQUIRY
    '\x06'     #  0x2E -> ACKNOWLEDGE
    '\x07'     #  0x2F -> BELL
    '\x90'     #  0x30 -> CONTROL
    '\x91'     #  0x31 -> CONTROL
    '\x16'     #  0x32 -> SYNCHRONOUS IDLE
    '\x93'     #  0x33 -> CONTROL
    '\x94'     #  0x34 -> CONTROL
    '\x95'     #  0x35 -> CONTROL
    '\x96'     #  0x36 -> CONTROL
    '\x04'     #  0x37 -> END OF TRANSMISSION
    '\x98'     #  0x38 -> CONTROL
    '\x99'     #  0x39 -> CONTROL
    '\x9a'     #  0x3A -> CONTROL
    '\x9b'     #  0x3B -> CONTROL
    '\x14'     #  0x3C -> DEVICE CONTROL FOUR
    '\x15'     #  0x3D -> NEGATIVE ACKNOWLEDGE
    '\x9e'     #  0x3E -> CONTROL
    '\x1a'     #  0x3F -> SUBSTITUTE
    ' '        #  0x40 -> SPACE
    '\u0391'   #  0x41 -> GREEK CAPITAL LETTER ALPHA
    '\u0392'   #  0x42 -> GREEK CAPITAL LETTER BETA
    '\u0393'   #  0x43 -> GREEK CAPITAL LETTER GAMMA
    '\u0394'   #  0x44 -> GREEK CAPITAL LETTER DELTA
    '\u0395'   #  0x45 -> GREEK CAPITAL LETTER EPSILON
    '\u0396'   #  0x46 -> GREEK CAPITAL LETTER ZETA
    '\u0397'   #  0x47 -> GREEK CAPITAL LETTER ETA
    '\u0398'   #  0x48 -> GREEK CAPITAL LETTER THETA
    '\u0399'   #  0x49 -> GREEK CAPITAL LETTER IOTA
    '['        #  0x4A -> LEFT SQUARE BRACKET
    '.'        #  0x4B -> FULL STOP
    '<'        #  0x4C -> LESS-THAN SIGN
    '('        #  0x4D -> LEFT PARENTHESIS
    '+'        #  0x4E -> PLUS SIGN
    '!'        #  0x4F -> EXCLAMATION MARK
    '&'        #  0x50 -> AMPERSAND
    '\u039a'   #  0x51 -> GREEK CAPITAL LETTER KAPPA
    '\u039b'   #  0x52 -> GREEK CAPITAL LETTER LAMDA
    '\u039c'   #  0x53 -> GREEK CAPITAL LETTER MU
    '\u039d'   #  0x54 -> GREEK CAPITAL LETTER NU
    '\u039e'   #  0x55 -> GREEK CAPITAL LETTER XI
    '\u039f'   #  0x56 -> GREEK CAPITAL LETTER OMICRON
    '\u03a0'   #  0x57 -> GREEK CAPITAL LETTER PI
    '\u03a1'   #  0x58 -> GREEK CAPITAL LETTER RHO
    '\u03a3'   #  0x59 -> GREEK CAPITAL LETTER SIGMA
    ']'        #  0x5A -> RIGHT SQUARE BRACKET
    '$'        #  0x5B -> DOLLAR SIGN
    '*'        #  0x5C -> ASTERISK
    ')'        #  0x5D -> RIGHT PARENTHESIS
    ';'        #  0x5E -> SEMICOLON
    '^'        #  0x5F -> CIRCUMFLEX ACCENT
    '-'        #  0x60 -> HYPHEN-MINUS
    '/'        #  0x61 -> SOLIDUS
    '\u03a4'   #  0x62 -> GREEK CAPITAL LETTER TAU
    '\u03a5'   #  0x63 -> GREEK CAPITAL LETTER UPSILON
    '\u03a6'   #  0x64 -> GREEK CAPITAL LETTER PHI
    '\u03a7'   #  0x65 -> GREEK CAPITAL LETTER CHI
    '\u03a8'   #  0x66 -> GREEK CAPITAL LETTER PSI
    '\u03a9'   #  0x67 -> GREEK CAPITAL LETTER OMEGA
    '\u03aa'   #  0x68 -> GREEK CAPITAL LETTER IOTA WITH DIALYTIKA
    '\u03ab'   #  0x69 -> GREEK CAPITAL LETTER UPSILON WITH DIALYTIKA
    '|'        #  0x6A -> VERTICAL LINE
    ','        #  0x6B -> COMMA
    '%'        #  0x6C -> PERCENT SIGN
    '_'        #  0x6D -> LOW LINE
    '>'        #  0x6E -> GREATER-THAN SIGN
    '?'        #  0x6F -> QUESTION MARK
    '\xa8'     #  0x70 -> DIAERESIS
    '\u0386'   #  0x71 -> GREEK CAPITAL LETTER ALPHA WITH TONOS
    '\u0388'   #  0x72 -> GREEK CAPITAL LETTER EPSILON WITH TONOS
    '\u0389'   #  0x73 -> GREEK CAPITAL LETTER ETA WITH TONOS
    '\xa0'     #  0x74 -> NO-BREAK SPACE
    '\u038a'   #  0x75 -> GREEK CAPITAL LETTER IOTA WITH TONOS
    '\u038c'   #  0x76 -> GREEK CAPITAL LETTER OMICRON WITH TONOS
    '\u038e'   #  0x77 -> GREEK CAPITAL LETTER UPSILON WITH TONOS
    '\u038f'   #  0x78 -> GREEK CAPITAL LETTER OMEGA WITH TONOS
    '`'        #  0x79 -> GRAVE ACCENT
    ':'        #  0x7A -> COLON
    '#'        #  0x7B -> NUMBER SIGN
    '@'        #  0x7C -> COMMERCIAL AT
    "'"        #  0x7D -> APOSTROPHE
    '='        #  0x7E -> EQUALS SIGN
    '"'        #  0x7F -> QUOTATION MARK
    '\u0385'   #  0x80 -> GREEK DIALYTIKA TONOS
    'a'        #  0x81 -> LATIN SMALL LETTER A
    'b'        #  0x82 -> LATIN SMALL LETTER B
    'c'        #  0x83 -> LATIN SMALL LETTER C
    'd'        #  0x84 -> LATIN SMALL LETTER D
    'e'        #  0x85 -> LATIN SMALL LETTER E
    'f'        #  0x86 -> LATIN SMALL LETTER F
    'g'        #  0x87 -> LATIN SMALL LETTER G
    'h'        #  0x88 -> LATIN SMALL LETTER H
    'i'        #  0x89 -> LATIN SMALL LETTER I
    '\u03b1'   #  0x8A -> GREEK SMALL LETTER ALPHA
    '\u03b2'   #  0x8B -> GREEK SMALL LETTER BETA
    '\u03b3'   #  0x8C -> GREEK SMALL LETTER GAMMA
    '\u03b4'   #  0x8D -> GREEK SMALL LETTER DELTA
    '\u03b5'   #  0x8E -> GREEK SMALL LETTER EPSILON
    '\u03b6'   #  0x8F -> GREEK SMALL LETTER ZETA
    '\xb0'     #  0x90 -> DEGREE SIGN
    'j'        #  0x91 -> LATIN SMALL LETTER J
    'k'        #  0x92 -> LATIN SMALL LETTER K
    'l'        #  0x93 -> LATIN SMALL LETTER L
    'm'        #  0x94 -> LATIN SMALL LETTER M
    'n'        #  0x95 -> LATIN SMALL LETTER N
    'o'        #  0x96 -> LATIN SMALL LETTER O
    'p'        #  0x97 -> LATIN SMALL LETTER P
    'q'        #  0x98 -> LATIN SMALL LETTER Q
    'r'        #  0x99 -> LATIN SMALL LETTER R
    '\u03b7'   #  0x9A -> GREEK SMALL LETTER ETA
    '\u03b8'   #  0x9B -> GREEK SMALL LETTER THETA
    '\u03b9'   #  0x9C -> GREEK SMALL LETTER IOTA
    '\u03ba'   #  0x9D -> GREEK SMALL LETTER KAPPA
    '\u03bb'   #  0x9E -> GREEK SMALL LETTER LAMDA
    '\u03bc'   #  0x9F -> GREEK SMALL LETTER MU
    '\xb4'     #  0xA0 -> ACUTE ACCENT
    '~'        #  0xA1 -> TILDE
    's'        #  0xA2 -> LATIN SMALL LETTER S
    't'        #  0xA3 -> LATIN SMALL LETTER T
    'u'        #  0xA4 -> LATIN SMALL LETTER U
    'v'        #  0xA5 -> LATIN SMALL LETTER V
    'w'        #  0xA6 -> LATIN SMALL LETTER W
    'x'        #  0xA7 -> LATIN SMALL LETTER X
    'y'        #  0xA8 -> LATIN SMALL LETTER Y
    'z'        #  0xA9 -> LATIN SMALL LETTER Z
    '\u03bd'   #  0xAA -> GREEK SMALL LETTER NU
    '\u03be'   #  0xAB -> GREEK SMALL LETTER XI
    '\u03bf'   #  0xAC -> GREEK SMALL LETTER OMICRON
    '\u03c0'   #  0xAD -> GREEK SMALL LETTER PI
    '\u03c1'   #  0xAE -> GREEK SMALL LETTER RHO
    '\u03c3'   #  0xAF -> GREEK SMALL LETTER SIGMA
    '\xa3'     #  0xB0 -> POUND SIGN
    '\u03ac'   #  0xB1 -> GREEK SMALL LETTER ALPHA WITH TONOS
    '\u03ad'   #  0xB2 -> GREEK SMALL LETTER EPSILON WITH TONOS
    '\u03ae'   #  0xB3 -> GREEK SMALL LETTER ETA WITH TONOS
    '\u03ca'   #  0xB4 -> GREEK SMALL LETTER IOTA WITH DIALYTIKA
    '\u03af'   #  0xB5 -> GREEK SMALL LETTER IOTA WITH TONOS
    '\u03cc'   #  0xB6 -> GREEK SMALL LETTER OMICRON WITH TONOS
    '\u03cd'   #  0xB7 -> GREEK SMALL LETTER UPSILON WITH TONOS
    '\u03cb'   #  0xB8 -> GREEK SMALL LETTER UPSILON WITH DIALYTIKA
    '\u03ce'   #  0xB9 -> GREEK SMALL LETTER OMEGA WITH TONOS
    '\u03c2'   #  0xBA -> GREEK SMALL LETTER FINAL SIGMA
    '\u03c4'   #  0xBB -> GREEK SMALL LETTER TAU
    '\u03c5'   #  0xBC -> GREEK SMALL LETTER UPSILON
    '\u03c6'   #  0xBD -> GREEK SMALL LETTER PHI
    '\u03c7'   #  0xBE -> GREEK SMALL LETTER CHI
    '\u03c8'   #  0xBF -> GREEK SMALL LETTER PSI
    '{'        #  0xC0 -> LEFT CURLY BRACKET
    'A'        #  0xC1 -> LATIN CAPITAL LETTER A
    'B'        #  0xC2 -> LATIN CAPITAL LETTER B
    'C'        #  0xC3 -> LATIN CAPITAL LETTER C
    'D'        #  0xC4 -> LATIN CAPITAL LETTER D
    'E'        #  0xC5 -> LATIN CAPITAL LETTER E
    'F'        #  0xC6 -> LATIN CAPITAL LETTER F
    'G'        #  0xC7 -> LATIN CAPITAL LETTER G
    'H'        #  0xC8 -> LATIN CAPITAL LETTER H
    'I'        #  0xC9 -> LATIN CAPITAL LETTER I
    '\xad'     #  0xCA -> SOFT HYPHEN
    '\u03c9'   #  0xCB -> GREEK SMALL LETTER OMEGA
    '\u0390'   #  0xCC -> GREEK SMALL LETTER IOTA WITH DIALYTIKA AND TONOS
    '\u03b0'   #  0xCD -> GREEK SMALL LETTER UPSILON WITH DIALYTIKA AND TONOS
    '\u2018'   #  0xCE -> LEFT SINGLE QUOTATION MARK
    '\u2015'   #  0xCF -> HORIZONTAL BAR
    '}'        #  0xD0 -> RIGHT CURLY BRACKET
    'J'        #  0xD1 -> LATIN CAPITAL LETTER J
    'K'        #  0xD2 -> LATIN CAPITAL LETTER K
    'L'        #  0xD3 -> LATIN CAPITAL LETTER L
    'M'        #  0xD4 -> LATIN CAPITAL LETTER M
    'N'        #  0xD5 -> LATIN CAPITAL LETTER N
    'O'        #  0xD6 -> LATIN CAPITAL LETTER O
    'P'        #  0xD7 -> LATIN CAPITAL LETTER P
    'Q'        #  0xD8 -> LATIN CAPITAL LETTER Q
    'R'        #  0xD9 -> LATIN CAPITAL LETTER R
    '\xb1'     #  0xDA -> PLUS-MINUS SIGN
    '\xbd'     #  0xDB -> VULGAR FRACTION ONE HALF
    '\x1a'     #  0xDC -> SUBSTITUTE
    '\u0387'   #  0xDD -> GREEK ANO TELEIA
    '\u2019'   #  0xDE -> RIGHT SINGLE QUOTATION MARK
    '\xa6'     #  0xDF -> BROKEN BAR
    '\\'       #  0xE0 -> REVERSE SOLIDUS
    '\x1a'     #  0xE1 -> SUBSTITUTE
    'S'        #  0xE2 -> LATIN CAPITAL LETTER S
    'T'        #  0xE3 -> LATIN CAPITAL LETTER T
    'U'        #  0xE4 -> LATIN CAPITAL LETTER U
    'V'        #  0xE5 -> LATIN CAPITAL LETTER V
    'W'        #  0xE6 -> LATIN CAPITAL LETTER W
    'X'        #  0xE7 -> LATIN CAPITAL LETTER X
    'Y'        #  0xE8 -> LATIN CAPITAL LETTER Y
    'Z'        #  0xE9 -> LATIN CAPITAL LETTER Z
    '\xb2'     #  0xEA -> SUPERSCRIPT TWO
    '\xa7'     #  0xEB -> SECTION SIGN
    '\x1a'     #  0xEC -> SUBSTITUTE
    '\x1a'     #  0xED -> SUBSTITUTE
    '\xab'     #  0xEE -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xac'     #  0xEF -> NOT SIGN
    '0'        #  0xF0 -> DIGIT ZERO
    '1'        #  0xF1 -> DIGIT ONE
    '2'        #  0xF2 -> DIGIT TWO
    '3'        #  0xF3 -> DIGIT THREE
    '4'        #  0xF4 -> DIGIT FOUR
    '5'        #  0xF5 -> DIGIT FIVE
    '6'        #  0xF6 -> DIGIT SIX
    '7'        #  0xF7 -> DIGIT SEVEN
    '8'        #  0xF8 -> DIGIT EIGHT
    '9'        #  0xF9 -> DIGIT NINE
    '\xb3'     #  0xFA -> SUPERSCRIPT THREE
    '\xa9'     #  0xFB -> COPYRIGHT SIGN
    '\x1a'     #  0xFC -> SUBSTITUTE
    '\x1a'     #  0xFD -> SUBSTITUTE
    '\xbb'     #  0xFE -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\x9f'     #  0xFF -> CONTROL
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
