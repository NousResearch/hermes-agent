""" Python Character Mapping Codec cp424 generated from 'MAPPINGS/VENDORS/MISC/CP424.TXT' with gencodec.py.

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
        name='cp424',
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
    '\x9c'     #  0x04 -> SELECT
    '\t'       #  0x05 -> HORIZONTAL TABULATION
    '\x86'     #  0x06 -> REQUIRED NEW LINE
    '\x7f'     #  0x07 -> DELETE
    '\x97'     #  0x08 -> GRAPHIC ESCAPE
    '\x8d'     #  0x09 -> SUPERSCRIPT
    '\x8e'     #  0x0A -> REPEAT
    '\x0b'     #  0x0B -> VERTICAL TABULATION
    '\x0c'     #  0x0C -> FORM FEED
    '\r'       #  0x0D -> CARRIAGE RETURN
    '\x0e'     #  0x0E -> SHIFT OUT
    '\x0f'     #  0x0F -> SHIFT IN
    '\x10'     #  0x10 -> DATA LINK ESCAPE
    '\x11'     #  0x11 -> DEVICE CONTROL ONE
    '\x12'     #  0x12 -> DEVICE CONTROL TWO
    '\x13'     #  0x13 -> DEVICE CONTROL THREE
    '\x9d'     #  0x14 -> RESTORE/ENABLE PRESENTATION
    '\x85'     #  0x15 -> NEW LINE
    '\x08'     #  0x16 -> BACKSPACE
    '\x87'     #  0x17 -> PROGRAM OPERATOR COMMUNICATION
    '\x18'     #  0x18 -> CANCEL
    '\x19'     #  0x19 -> END OF MEDIUM
    '\x92'     #  0x1A -> UNIT BACK SPACE
    '\x8f'     #  0x1B -> CUSTOMER USE ONE
    '\x1c'     #  0x1C -> FILE SEPARATOR
    '\x1d'     #  0x1D -> GROUP SEPARATOR
    '\x1e'     #  0x1E -> RECORD SEPARATOR
    '\x1f'     #  0x1F -> UNIT SEPARATOR
    '\x80'     #  0x20 -> DIGIT SELECT
    '\x81'     #  0x21 -> START OF SIGNIFICANCE
    '\x82'     #  0x22 -> FIELD SEPARATOR
    '\x83'     #  0x23 -> WORD UNDERSCORE
    '\x84'     #  0x24 -> BYPASS OR INHIBIT PRESENTATION
    '\n'       #  0x25 -> LINE FEED
    '\x17'     #  0x26 -> END OF TRANSMISSION BLOCK
    '\x1b'     #  0x27 -> ESCAPE
    '\x88'     #  0x28 -> SET ATTRIBUTE
    '\x89'     #  0x29 -> START FIELD EXTENDED
    '\x8a'     #  0x2A -> SET MODE OR SWITCH
    '\x8b'     #  0x2B -> CONTROL SEQUENCE PREFIX
    '\x8c'     #  0x2C -> MODIFY FIELD ATTRIBUTE
    '\x05'     #  0x2D -> ENQUIRY
    '\x06'     #  0x2E -> ACKNOWLEDGE
    '\x07'     #  0x2F -> BELL
    '\x90'     #  0x30 -> <reserved>
    '\x91'     #  0x31 -> <reserved>
    '\x16'     #  0x32 -> SYNCHRONOUS IDLE
    '\x93'     #  0x33 -> INDEX RETURN
    '\x94'     #  0x34 -> PRESENTATION POSITION
    '\x95'     #  0x35 -> TRANSPARENT
    '\x96'     #  0x36 -> NUMERIC BACKSPACE
    '\x04'     #  0x37 -> END OF TRANSMISSION
    '\x98'     #  0x38 -> SUBSCRIPT
    '\x99'     #  0x39 -> INDENT TABULATION
    '\x9a'     #  0x3A -> REVERSE FORM FEED
    '\x9b'     #  0x3B -> CUSTOMER USE THREE
    '\x14'     #  0x3C -> DEVICE CONTROL FOUR
    '\x15'     #  0x3D -> NEGATIVE ACKNOWLEDGE
    '\x9e'     #  0x3E -> <reserved>
    '\x1a'     #  0x3F -> SUBSTITUTE
    ' '        #  0x40 -> SPACE
    '\u05d0'   #  0x41 -> HEBREW LETTER ALEF
    '\u05d1'   #  0x42 -> HEBREW LETTER BET
    '\u05d2'   #  0x43 -> HEBREW LETTER GIMEL
    '\u05d3'   #  0x44 -> HEBREW LETTER DALET
    '\u05d4'   #  0x45 -> HEBREW LETTER HE
    '\u05d5'   #  0x46 -> HEBREW LETTER VAV
    '\u05d6'   #  0x47 -> HEBREW LETTER ZAYIN
    '\u05d7'   #  0x48 -> HEBREW LETTER HET
    '\u05d8'   #  0x49 -> HEBREW LETTER TET
    '\xa2'     #  0x4A -> CENT SIGN
    '.'        #  0x4B -> FULL STOP
    '<'        #  0x4C -> LESS-THAN SIGN
    '('        #  0x4D -> LEFT PARENTHESIS
    '+'        #  0x4E -> PLUS SIGN
    '|'        #  0x4F -> VERTICAL LINE
    '&'        #  0x50 -> AMPERSAND
    '\u05d9'   #  0x51 -> HEBREW LETTER YOD
    '\u05da'   #  0x52 -> HEBREW LETTER FINAL KAF
    '\u05db'   #  0x53 -> HEBREW LETTER KAF
    '\u05dc'   #  0x54 -> HEBREW LETTER LAMED
    '\u05dd'   #  0x55 -> HEBREW LETTER FINAL MEM
    '\u05de'   #  0x56 -> HEBREW LETTER MEM
    '\u05df'   #  0x57 -> HEBREW LETTER FINAL NUN
    '\u05e0'   #  0x58 -> HEBREW LETTER NUN
    '\u05e1'   #  0x59 -> HEBREW LETTER SAMEKH
    '!'        #  0x5A -> EXCLAMATION MARK
    '$'        #  0x5B -> DOLLAR SIGN
    '*'        #  0x5C -> ASTERISK
    ')'        #  0x5D -> RIGHT PARENTHESIS
    ';'        #  0x5E -> SEMICOLON
    '\xac'     #  0x5F -> NOT SIGN
    '-'        #  0x60 -> HYPHEN-MINUS
    '/'        #  0x61 -> SOLIDUS
    '\u05e2'   #  0x62 -> HEBREW LETTER AYIN
    '\u05e3'   #  0x63 -> HEBREW LETTER FINAL PE
    '\u05e4'   #  0x64 -> HEBREW LETTER PE
    '\u05e5'   #  0x65 -> HEBREW LETTER FINAL TSADI
    '\u05e6'   #  0x66 -> HEBREW LETTER TSADI
    '\u05e7'   #  0x67 -> HEBREW LETTER QOF
    '\u05e8'   #  0x68 -> HEBREW LETTER RESH
    '\u05e9'   #  0x69 -> HEBREW LETTER SHIN
    '\xa6'     #  0x6A -> BROKEN BAR
    ','        #  0x6B -> COMMA
    '%'        #  0x6C -> PERCENT SIGN
    '_'        #  0x6D -> LOW LINE
    '>'        #  0x6E -> GREATER-THAN SIGN
    '?'        #  0x6F -> QUESTION MARK
    '\ufffe'   #  0x70 -> UNDEFINED
    '\u05ea'   #  0x71 -> HEBREW LETTER TAV
    '\ufffe'   #  0x72 -> UNDEFINED
    '\ufffe'   #  0x73 -> UNDEFINED
    '\xa0'     #  0x74 -> NO-BREAK SPACE
    '\ufffe'   #  0x75 -> UNDEFINED
    '\ufffe'   #  0x76 -> UNDEFINED
    '\ufffe'   #  0x77 -> UNDEFINED
    '\u2017'   #  0x78 -> DOUBLE LOW LINE
    '`'        #  0x79 -> GRAVE ACCENT
    ':'        #  0x7A -> COLON
    '#'        #  0x7B -> NUMBER SIGN
    '@'        #  0x7C -> COMMERCIAL AT
    "'"        #  0x7D -> APOSTROPHE
    '='        #  0x7E -> EQUALS SIGN
    '"'        #  0x7F -> QUOTATION MARK
    '\ufffe'   #  0x80 -> UNDEFINED
    'a'        #  0x81 -> LATIN SMALL LETTER A
    'b'        #  0x82 -> LATIN SMALL LETTER B
    'c'        #  0x83 -> LATIN SMALL LETTER C
    'd'        #  0x84 -> LATIN SMALL LETTER D
    'e'        #  0x85 -> LATIN SMALL LETTER E
    'f'        #  0x86 -> LATIN SMALL LETTER F
    'g'        #  0x87 -> LATIN SMALL LETTER G
    'h'        #  0x88 -> LATIN SMALL LETTER H
    'i'        #  0x89 -> LATIN SMALL LETTER I
    '\xab'     #  0x8A -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbb'     #  0x8B -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\ufffe'   #  0x8C -> UNDEFINED
    '\ufffe'   #  0x8D -> UNDEFINED
    '\ufffe'   #  0x8E -> UNDEFINED
    '\xb1'     #  0x8F -> PLUS-MINUS SIGN
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
    '\ufffe'   #  0x9A -> UNDEFINED
    '\ufffe'   #  0x9B -> UNDEFINED
    '\ufffe'   #  0x9C -> UNDEFINED
    '\xb8'     #  0x9D -> CEDILLA
    '\ufffe'   #  0x9E -> UNDEFINED
    '\xa4'     #  0x9F -> CURRENCY SIGN
    '\xb5'     #  0xA0 -> MICRO SIGN
    '~'        #  0xA1 -> TILDE
    's'        #  0xA2 -> LATIN SMALL LETTER S
    't'        #  0xA3 -> LATIN SMALL LETTER T
    'u'        #  0xA4 -> LATIN SMALL LETTER U
    'v'        #  0xA5 -> LATIN SMALL LETTER V
    'w'        #  0xA6 -> LATIN SMALL LETTER W
    'x'        #  0xA7 -> LATIN SMALL LETTER X
    'y'        #  0xA8 -> LATIN SMALL LETTER Y
    'z'        #  0xA9 -> LATIN SMALL LETTER Z
    '\ufffe'   #  0xAA -> UNDEFINED
    '\ufffe'   #  0xAB -> UNDEFINED
    '\ufffe'   #  0xAC -> UNDEFINED
    '\ufffe'   #  0xAD -> UNDEFINED
    '\ufffe'   #  0xAE -> UNDEFINED
    '\xae'     #  0xAF -> REGISTERED SIGN
    '^'        #  0xB0 -> CIRCUMFLEX ACCENT
    '\xa3'     #  0xB1 -> POUND SIGN
    '\xa5'     #  0xB2 -> YEN SIGN
    '\xb7'     #  0xB3 -> MIDDLE DOT
    '\xa9'     #  0xB4 -> COPYRIGHT SIGN
    '\xa7'     #  0xB5 -> SECTION SIGN
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\xbc'     #  0xB7 -> VULGAR FRACTION ONE QUARTER
    '\xbd'     #  0xB8 -> VULGAR FRACTION ONE HALF
    '\xbe'     #  0xB9 -> VULGAR FRACTION THREE QUARTERS
    '['        #  0xBA -> LEFT SQUARE BRACKET
    ']'        #  0xBB -> RIGHT SQUARE BRACKET
    '\xaf'     #  0xBC -> MACRON
    '\xa8'     #  0xBD -> DIAERESIS
    '\xb4'     #  0xBE -> ACUTE ACCENT
    '\xd7'     #  0xBF -> MULTIPLICATION SIGN
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
    '\ufffe'   #  0xCB -> UNDEFINED
    '\ufffe'   #  0xCC -> UNDEFINED
    '\ufffe'   #  0xCD -> UNDEFINED
    '\ufffe'   #  0xCE -> UNDEFINED
    '\ufffe'   #  0xCF -> UNDEFINED
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
    '\xb9'     #  0xDA -> SUPERSCRIPT ONE
    '\ufffe'   #  0xDB -> UNDEFINED
    '\ufffe'   #  0xDC -> UNDEFINED
    '\ufffe'   #  0xDD -> UNDEFINED
    '\ufffe'   #  0xDE -> UNDEFINED
    '\ufffe'   #  0xDF -> UNDEFINED
    '\\'       #  0xE0 -> REVERSE SOLIDUS
    '\xf7'     #  0xE1 -> DIVISION SIGN
    'S'        #  0xE2 -> LATIN CAPITAL LETTER S
    'T'        #  0xE3 -> LATIN CAPITAL LETTER T
    'U'        #  0xE4 -> LATIN CAPITAL LETTER U
    'V'        #  0xE5 -> LATIN CAPITAL LETTER V
    'W'        #  0xE6 -> LATIN CAPITAL LETTER W
    'X'        #  0xE7 -> LATIN CAPITAL LETTER X
    'Y'        #  0xE8 -> LATIN CAPITAL LETTER Y
    'Z'        #  0xE9 -> LATIN CAPITAL LETTER Z
    '\xb2'     #  0xEA -> SUPERSCRIPT TWO
    '\ufffe'   #  0xEB -> UNDEFINED
    '\ufffe'   #  0xEC -> UNDEFINED
    '\ufffe'   #  0xED -> UNDEFINED
    '\ufffe'   #  0xEE -> UNDEFINED
    '\ufffe'   #  0xEF -> UNDEFINED
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
    '\ufffe'   #  0xFB -> UNDEFINED
    '\ufffe'   #  0xFC -> UNDEFINED
    '\ufffe'   #  0xFD -> UNDEFINED
    '\ufffe'   #  0xFE -> UNDEFINED
    '\x9f'     #  0xFF -> EIGHT ONES
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
