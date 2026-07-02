""" Python Character Mapping Codec koi8_t
"""
# http://ru.wikipedia.org/wiki/КОИ-8
# http://www.opensource.apple.com/source/libiconv/libiconv-4/libiconv/tests/KOI8-T.TXT

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
        name='koi8-t',
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
    '\x04'     #  0x04 -> END OF TRANSMISSION
    '\x05'     #  0x05 -> ENQUIRY
    '\x06'     #  0x06 -> ACKNOWLEDGE
    '\x07'     #  0x07 -> BELL
    '\x08'     #  0x08 -> BACKSPACE
    '\t'       #  0x09 -> HORIZONTAL TABULATION
    '\n'       #  0x0A -> LINE FEED
    '\x0b'     #  0x0B -> VERTICAL TABULATION
    '\x0c'     #  0x0C -> FORM FEED
    '\r'       #  0x0D -> CARRIAGE RETURN
    '\x0e'     #  0x0E -> SHIFT OUT
    '\x0f'     #  0x0F -> SHIFT IN
    '\x10'     #  0x10 -> DATA LINK ESCAPE
    '\x11'     #  0x11 -> DEVICE CONTROL ONE
    '\x12'     #  0x12 -> DEVICE CONTROL TWO
    '\x13'     #  0x13 -> DEVICE CONTROL THREE
    '\x14'     #  0x14 -> DEVICE CONTROL FOUR
    '\x15'     #  0x15 -> NEGATIVE ACKNOWLEDGE
    '\x16'     #  0x16 -> SYNCHRONOUS IDLE
    '\x17'     #  0x17 -> END OF TRANSMISSION BLOCK
    '\x18'     #  0x18 -> CANCEL
    '\x19'     #  0x19 -> END OF MEDIUM
    '\x1a'     #  0x1A -> SUBSTITUTE
    '\x1b'     #  0x1B -> ESCAPE
    '\x1c'     #  0x1C -> FILE SEPARATOR
    '\x1d'     #  0x1D -> GROUP SEPARATOR
    '\x1e'     #  0x1E -> RECORD SEPARATOR
    '\x1f'     #  0x1F -> UNIT SEPARATOR
    ' '        #  0x20 -> SPACE
    '!'        #  0x21 -> EXCLAMATION MARK
    '"'        #  0x22 -> QUOTATION MARK
    '#'        #  0x23 -> NUMBER SIGN
    '$'        #  0x24 -> DOLLAR SIGN
    '%'        #  0x25 -> PERCENT SIGN
    '&'        #  0x26 -> AMPERSAND
    "'"        #  0x27 -> APOSTROPHE
    '('        #  0x28 -> LEFT PARENTHESIS
    ')'        #  0x29 -> RIGHT PARENTHESIS
    '*'        #  0x2A -> ASTERISK
    '+'        #  0x2B -> PLUS SIGN
    ','        #  0x2C -> COMMA
    '-'        #  0x2D -> HYPHEN-MINUS
    '.'        #  0x2E -> FULL STOP
    '/'        #  0x2F -> SOLIDUS
    '0'        #  0x30 -> DIGIT ZERO
    '1'        #  0x31 -> DIGIT ONE
    '2'        #  0x32 -> DIGIT TWO
    '3'        #  0x33 -> DIGIT THREE
    '4'        #  0x34 -> DIGIT FOUR
    '5'        #  0x35 -> DIGIT FIVE
    '6'        #  0x36 -> DIGIT SIX
    '7'        #  0x37 -> DIGIT SEVEN
    '8'        #  0x38 -> DIGIT EIGHT
    '9'        #  0x39 -> DIGIT NINE
    ':'        #  0x3A -> COLON
    ';'        #  0x3B -> SEMICOLON
    '<'        #  0x3C -> LESS-THAN SIGN
    '='        #  0x3D -> EQUALS SIGN
    '>'        #  0x3E -> GREATER-THAN SIGN
    '?'        #  0x3F -> QUESTION MARK
    '@'        #  0x40 -> COMMERCIAL AT
    'A'        #  0x41 -> LATIN CAPITAL LETTER A
    'B'        #  0x42 -> LATIN CAPITAL LETTER B
    'C'        #  0x43 -> LATIN CAPITAL LETTER C
    'D'        #  0x44 -> LATIN CAPITAL LETTER D
    'E'        #  0x45 -> LATIN CAPITAL LETTER E
    'F'        #  0x46 -> LATIN CAPITAL LETTER F
    'G'        #  0x47 -> LATIN CAPITAL LETTER G
    'H'        #  0x48 -> LATIN CAPITAL LETTER H
    'I'        #  0x49 -> LATIN CAPITAL LETTER I
    'J'        #  0x4A -> LATIN CAPITAL LETTER J
    'K'        #  0x4B -> LATIN CAPITAL LETTER K
    'L'        #  0x4C -> LATIN CAPITAL LETTER L
    'M'        #  0x4D -> LATIN CAPITAL LETTER M
    'N'        #  0x4E -> LATIN CAPITAL LETTER N
    'O'        #  0x4F -> LATIN CAPITAL LETTER O
    'P'        #  0x50 -> LATIN CAPITAL LETTER P
    'Q'        #  0x51 -> LATIN CAPITAL LETTER Q
    'R'        #  0x52 -> LATIN CAPITAL LETTER R
    'S'        #  0x53 -> LATIN CAPITAL LETTER S
    'T'        #  0x54 -> LATIN CAPITAL LETTER T
    'U'        #  0x55 -> LATIN CAPITAL LETTER U
    'V'        #  0x56 -> LATIN CAPITAL LETTER V
    'W'        #  0x57 -> LATIN CAPITAL LETTER W
    'X'        #  0x58 -> LATIN CAPITAL LETTER X
    'Y'        #  0x59 -> LATIN CAPITAL LETTER Y
    'Z'        #  0x5A -> LATIN CAPITAL LETTER Z
    '['        #  0x5B -> LEFT SQUARE BRACKET
    '\\'       #  0x5C -> REVERSE SOLIDUS
    ']'        #  0x5D -> RIGHT SQUARE BRACKET
    '^'        #  0x5E -> CIRCUMFLEX ACCENT
    '_'        #  0x5F -> LOW LINE
    '`'        #  0x60 -> GRAVE ACCENT
    'a'        #  0x61 -> LATIN SMALL LETTER A
    'b'        #  0x62 -> LATIN SMALL LETTER B
    'c'        #  0x63 -> LATIN SMALL LETTER C
    'd'        #  0x64 -> LATIN SMALL LETTER D
    'e'        #  0x65 -> LATIN SMALL LETTER E
    'f'        #  0x66 -> LATIN SMALL LETTER F
    'g'        #  0x67 -> LATIN SMALL LETTER G
    'h'        #  0x68 -> LATIN SMALL LETTER H
    'i'        #  0x69 -> LATIN SMALL LETTER I
    'j'        #  0x6A -> LATIN SMALL LETTER J
    'k'        #  0x6B -> LATIN SMALL LETTER K
    'l'        #  0x6C -> LATIN SMALL LETTER L
    'm'        #  0x6D -> LATIN SMALL LETTER M
    'n'        #  0x6E -> LATIN SMALL LETTER N
    'o'        #  0x6F -> LATIN SMALL LETTER O
    'p'        #  0x70 -> LATIN SMALL LETTER P
    'q'        #  0x71 -> LATIN SMALL LETTER Q
    'r'        #  0x72 -> LATIN SMALL LETTER R
    's'        #  0x73 -> LATIN SMALL LETTER S
    't'        #  0x74 -> LATIN SMALL LETTER T
    'u'        #  0x75 -> LATIN SMALL LETTER U
    'v'        #  0x76 -> LATIN SMALL LETTER V
    'w'        #  0x77 -> LATIN SMALL LETTER W
    'x'        #  0x78 -> LATIN SMALL LETTER X
    'y'        #  0x79 -> LATIN SMALL LETTER Y
    'z'        #  0x7A -> LATIN SMALL LETTER Z
    '{'        #  0x7B -> LEFT CURLY BRACKET
    '|'        #  0x7C -> VERTICAL LINE
    '}'        #  0x7D -> RIGHT CURLY BRACKET
    '~'        #  0x7E -> TILDE
    '\x7f'     #  0x7F -> DELETE
    '\u049b'   #  0x80 -> CYRILLIC SMALL LETTER KA WITH DESCENDER
    '\u0493'   #  0x81 -> CYRILLIC SMALL LETTER GHE WITH STROKE
    '\u201a'   #  0x82 -> SINGLE LOW-9 QUOTATION MARK
    '\u0492'   #  0x83 -> CYRILLIC CAPITAL LETTER GHE WITH STROKE
    '\u201e'   #  0x84 -> DOUBLE LOW-9 QUOTATION MARK
    '\u2026'   #  0x85 -> HORIZONTAL ELLIPSIS
    '\u2020'   #  0x86 -> DAGGER
    '\u2021'   #  0x87 -> DOUBLE DAGGER
    '\ufffe'   #  0x88 -> UNDEFINED
    '\u2030'   #  0x89 -> PER MILLE SIGN
    '\u04b3'   #  0x8A -> CYRILLIC SMALL LETTER HA WITH DESCENDER
    '\u2039'   #  0x8B -> SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    '\u04b2'   #  0x8C -> CYRILLIC CAPITAL LETTER HA WITH DESCENDER
    '\u04b7'   #  0x8D -> CYRILLIC SMALL LETTER CHE WITH DESCENDER
    '\u04b6'   #  0x8E -> CYRILLIC CAPITAL LETTER CHE WITH DESCENDER
    '\ufffe'   #  0x8F -> UNDEFINED
    '\u049a'   #  0x90 -> CYRILLIC CAPITAL LETTER KA WITH DESCENDER
    '\u2018'   #  0x91 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0x92 -> RIGHT SINGLE QUOTATION MARK
    '\u201c'   #  0x93 -> LEFT DOUBLE QUOTATION MARK
    '\u201d'   #  0x94 -> RIGHT DOUBLE QUOTATION MARK
    '\u2022'   #  0x95 -> BULLET
    '\u2013'   #  0x96 -> EN DASH
    '\u2014'   #  0x97 -> EM DASH
    '\ufffe'   #  0x98 -> UNDEFINED
    '\u2122'   #  0x99 -> TRADE MARK SIGN
    '\ufffe'   #  0x9A -> UNDEFINED
    '\u203a'   #  0x9B -> SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    '\ufffe'   #  0x9C -> UNDEFINED
    '\ufffe'   #  0x9D -> UNDEFINED
    '\ufffe'   #  0x9E -> UNDEFINED
    '\ufffe'   #  0x9F -> UNDEFINED
    '\ufffe'   #  0xA0 -> UNDEFINED
    '\u04ef'   #  0xA1 -> CYRILLIC SMALL LETTER U WITH MACRON
    '\u04ee'   #  0xA2 -> CYRILLIC CAPITAL LETTER U WITH MACRON
    '\u0451'   #  0xA3 -> CYRILLIC SMALL LETTER IO
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\u04e3'   #  0xA5 -> CYRILLIC SMALL LETTER I WITH MACRON
    '\xa6'     #  0xA6 -> BROKEN BAR
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\ufffe'   #  0xA8 -> UNDEFINED
    '\ufffe'   #  0xA9 -> UNDEFINED
    '\ufffe'   #  0xAA -> UNDEFINED
    '\xab'     #  0xAB -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xac'     #  0xAC -> NOT SIGN
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\xae'     #  0xAE -> REGISTERED SIGN
    '\ufffe'   #  0xAF -> UNDEFINED
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\xb2'     #  0xB2 -> SUPERSCRIPT TWO
    '\u0401'   #  0xB3 -> CYRILLIC CAPITAL LETTER IO
    '\ufffe'   #  0xB4 -> UNDEFINED
    '\u04e2'   #  0xB5 -> CYRILLIC CAPITAL LETTER I WITH MACRON
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\xb7'     #  0xB7 -> MIDDLE DOT
    '\ufffe'   #  0xB8 -> UNDEFINED
    '\u2116'   #  0xB9 -> NUMERO SIGN
    '\ufffe'   #  0xBA -> UNDEFINED
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\ufffe'   #  0xBC -> UNDEFINED
    '\ufffe'   #  0xBD -> UNDEFINED
    '\ufffe'   #  0xBE -> UNDEFINED
    '\xa9'     #  0xBF -> COPYRIGHT SIGN
    '\u044e'   #  0xC0 -> CYRILLIC SMALL LETTER YU
    '\u0430'   #  0xC1 -> CYRILLIC SMALL LETTER A
    '\u0431'   #  0xC2 -> CYRILLIC SMALL LETTER BE
    '\u0446'   #  0xC3 -> CYRILLIC SMALL LETTER TSE
    '\u0434'   #  0xC4 -> CYRILLIC SMALL LETTER DE
    '\u0435'   #  0xC5 -> CYRILLIC SMALL LETTER IE
    '\u0444'   #  0xC6 -> CYRILLIC SMALL LETTER EF
    '\u0433'   #  0xC7 -> CYRILLIC SMALL LETTER GHE
    '\u0445'   #  0xC8 -> CYRILLIC SMALL LETTER HA
    '\u0438'   #  0xC9 -> CYRILLIC SMALL LETTER I
    '\u0439'   #  0xCA -> CYRILLIC SMALL LETTER SHORT I
    '\u043a'   #  0xCB -> CYRILLIC SMALL LETTER KA
    '\u043b'   #  0xCC -> CYRILLIC SMALL LETTER EL
    '\u043c'   #  0xCD -> CYRILLIC SMALL LETTER EM
    '\u043d'   #  0xCE -> CYRILLIC SMALL LETTER EN
    '\u043e'   #  0xCF -> CYRILLIC SMALL LETTER O
    '\u043f'   #  0xD0 -> CYRILLIC SMALL LETTER PE
    '\u044f'   #  0xD1 -> CYRILLIC SMALL LETTER YA
    '\u0440'   #  0xD2 -> CYRILLIC SMALL LETTER ER
    '\u0441'   #  0xD3 -> CYRILLIC SMALL LETTER ES
    '\u0442'   #  0xD4 -> CYRILLIC SMALL LETTER TE
    '\u0443'   #  0xD5 -> CYRILLIC SMALL LETTER U
    '\u0436'   #  0xD6 -> CYRILLIC SMALL LETTER ZHE
    '\u0432'   #  0xD7 -> CYRILLIC SMALL LETTER VE
    '\u044c'   #  0xD8 -> CYRILLIC SMALL LETTER SOFT SIGN
    '\u044b'   #  0xD9 -> CYRILLIC SMALL LETTER YERU
    '\u0437'   #  0xDA -> CYRILLIC SMALL LETTER ZE
    '\u0448'   #  0xDB -> CYRILLIC SMALL LETTER SHA
    '\u044d'   #  0xDC -> CYRILLIC SMALL LETTER E
    '\u0449'   #  0xDD -> CYRILLIC SMALL LETTER SHCHA
    '\u0447'   #  0xDE -> CYRILLIC SMALL LETTER CHE
    '\u044a'   #  0xDF -> CYRILLIC SMALL LETTER HARD SIGN
    '\u042e'   #  0xE0 -> CYRILLIC CAPITAL LETTER YU
    '\u0410'   #  0xE1 -> CYRILLIC CAPITAL LETTER A
    '\u0411'   #  0xE2 -> CYRILLIC CAPITAL LETTER BE
    '\u0426'   #  0xE3 -> CYRILLIC CAPITAL LETTER TSE
    '\u0414'   #  0xE4 -> CYRILLIC CAPITAL LETTER DE
    '\u0415'   #  0xE5 -> CYRILLIC CAPITAL LETTER IE
    '\u0424'   #  0xE6 -> CYRILLIC CAPITAL LETTER EF
    '\u0413'   #  0xE7 -> CYRILLIC CAPITAL LETTER GHE
    '\u0425'   #  0xE8 -> CYRILLIC CAPITAL LETTER HA
    '\u0418'   #  0xE9 -> CYRILLIC CAPITAL LETTER I
    '\u0419'   #  0xEA -> CYRILLIC CAPITAL LETTER SHORT I
    '\u041a'   #  0xEB -> CYRILLIC CAPITAL LETTER KA
    '\u041b'   #  0xEC -> CYRILLIC CAPITAL LETTER EL
    '\u041c'   #  0xED -> CYRILLIC CAPITAL LETTER EM
    '\u041d'   #  0xEE -> CYRILLIC CAPITAL LETTER EN
    '\u041e'   #  0xEF -> CYRILLIC CAPITAL LETTER O
    '\u041f'   #  0xF0 -> CYRILLIC CAPITAL LETTER PE
    '\u042f'   #  0xF1 -> CYRILLIC CAPITAL LETTER YA
    '\u0420'   #  0xF2 -> CYRILLIC CAPITAL LETTER ER
    '\u0421'   #  0xF3 -> CYRILLIC CAPITAL LETTER ES
    '\u0422'   #  0xF4 -> CYRILLIC CAPITAL LETTER TE
    '\u0423'   #  0xF5 -> CYRILLIC CAPITAL LETTER U
    '\u0416'   #  0xF6 -> CYRILLIC CAPITAL LETTER ZHE
    '\u0412'   #  0xF7 -> CYRILLIC CAPITAL LETTER VE
    '\u042c'   #  0xF8 -> CYRILLIC CAPITAL LETTER SOFT SIGN
    '\u042b'   #  0xF9 -> CYRILLIC CAPITAL LETTER YERU
    '\u0417'   #  0xFA -> CYRILLIC CAPITAL LETTER ZE
    '\u0428'   #  0xFB -> CYRILLIC CAPITAL LETTER SHA
    '\u042d'   #  0xFC -> CYRILLIC CAPITAL LETTER E
    '\u0429'   #  0xFD -> CYRILLIC CAPITAL LETTER SHCHA
    '\u0427'   #  0xFE -> CYRILLIC CAPITAL LETTER CHE
    '\u042a'   #  0xFF -> CYRILLIC CAPITAL LETTER HARD SIGN
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
