"""Provide advanced parsing abilities for ParenMatch and other extensions.

HyperParser uses PyParser.  PyParser mostly gives information on the
proper indentation of code.  HyperParser gives additional information on
the structure of code.
"""
from keyword import iskeyword
import string

from idlelib import pyparse

# all ASCII chars that may be in an identifier
_ASCII_ID_CHARS = frozenset(string.ascii_letters + string.digits + "_")
# all ASCII chars that may be the first char of an identifier
_ASCII_ID_FIRST_CHARS = frozenset(string.ascii_letters + "_")

# lookup table for whether 7-bit ASCII chars are valid in a Python identifier
_IS_ASCII_ID_CHAR = [(chr(x) in _ASCII_ID_CHARS) for x in range(128)]
# lookup table for whether 7-bit ASCII chars are valid as the first
# char in a Python identifier
_IS_ASCII_ID_FIRST_CHAR = \
    [(chr(x) in _ASCII_ID_FIRST_CHARS) for x in range(128)]


class HyperParser:
    def __init__(self, editwin, index):
        "To initialize, analyze the surroundings of the given index."

        self.editwin = editwin
        self.text = text = editwin.text

        parser = pyparse.Parser(editwin.indentwidth, editwin.tabwidth)

        def index2line(index):
            return int(float(index))
        lno = index2line(text.index(index))

        if not editwin.prompt_last_line:
            for context in editwin.num_context_lines:
                startat = max(lno - context, 1)
                startatindex = repr(startat) + ".0"
                stopatindex = "%d.end" % lno
                # We add the newline because PyParse requires a newline
                # at end. We add a space so that index won't be at end
                # of line, so that its status will be the same as the
                # char before it, if should.
                parser.set_code(text.get(startatindex, stopatindex)+' \n')
                bod = parser.find_good_parse_start(
                          editwin._build_char_in_string_func(startatindex))
                if bod is not None or startat == 1:
                    break
            parser.set_lo(bod or 0)
        else:
            r = text.tag_prevrange("console", index)
            if r:
                startatindex = r[1]
            else:
                startatindex = "1.0"
            stopatindex = "%d.end" % lno
            # We add the newline because PyParse requires it. We add a
            # space so that index won't be at end of line, so that its
            # status will be the same as the char before it, if should.
            parser.set_code(text.get(startatindex, stopatindex)+' \n')
            parser.set_lo(0)

        # We want what the parser has, minus the last newline and space.
        self.rawtext = parser.code[:-2]
        # Parser.code apparently preserves the statement we are in, so
        # that stopatindex can be used to synchronize the string with
        # the text box indices.
        self.stopatindex = stopatindex
        self.bracketing = parser.get_last_stmt_bracketing()
        # find which pairs of bracketing are openers. These always
        # correspond to a character of rawtext.
        self.isopener = [i>0 and self.bracketing[i][1] >
                         self.bracketing[i-1][1]
                         for i in range(len(self.bracketing))]

        self.set_index(index)

    def set_index(self, index):
        """Set the index to which the functions relate.

        The index must be in the same statement.
        """
        indexinrawtext = (len(self.rawtext) -
                          len(self.text.get(index, self.stopatindex)))
        if indexinrawtext < 0:
            raise ValueError("Index %s precedes the analyzed statement"
                             % index)
        self.indexinrawtext = indexinrawtext
        # find the rightmost bracket to which index belongs
        self.indexbracket = 0
        while (self.indexbracket < len(self.bracketing)-1 and
               self.bracketing[self.indexbracket+1][0] < self.indexinrawtext):
            self.indexbracket += 1
        if (self.indexbracket < len(self.bracketing)-1 and
            self.bracketing[self.indexbracket+1][0] == self.indexinrawtext and
           not self.isopener[self.indexbracket+1]):
            self.indexbracket += 1

    def is_in_string(self):
        """Is the index given to the HyperParser in a string?"""
        # The bracket to which we belong should be an opener.
        # If it's an opener, it has to have a character.
        return (self.isopener[self.indexbracket] and
                self.rawtext[self.bracketing[self.indexbracket][0]]
                in ('"', "'"))

    def is_in_code(self):
        """Is the index given to the HyperParser in normal code?"""
        return (not self.isopener[self.indexbracket] or
                self.rawtext[self.bracketing[self.indexbracket][0]]
                not in ('#', '"', "'"))

    def get_surrounding_brackets(self, openers='([{', mustclose=False):
        """Return bracket indexes or None.

        If the index given to the HyperParser is surrounded by a
        bracket defined in openers (or at least has one before it),
        return the indices of the opening bracket and the closing
        bracket (or the end of line, whichever comes first).

        If it is not surrounded by brackets, or the end of line comes
        before the closing bracket and mustclose is True, returns None.
        """

        bracketinglevel = self.bracketing[self.indexbracket][1]
        before = self.indexbracket
        while (not self.isopener[before] or
              self.rawtext[self.bracketing[before][0]] not in openers or
              self.bracketing[before][1] > bracketinglevel):
            before -= 1
            if before < 0:
                return None
            bracketinglevel = min(bracketinglevel, self.bracketing[before][1])
        after = self.indexbracket + 1
        while (after < len(self.bracketing) and
              self.bracketing[after][1] >= bracketinglevel):
            after += 1

        beforeindex = self.text.index("%s-%dc" %
            (self.stopatindex, len(self.rawtext)-self.bracketing[before][0]))
        if (after >= len(self.bracketing) or
           self.bracketing[after][0] > len(self.rawtext)):
            if mustclose:
                return None
            afterindex = self.stopatindex
        else:
            # We are after a real char, so it is a ')' and we give the
            # index before it.
            afterindex = self.text.index(
                "%s-%dc" % (self.stopatindex,
                 len(self.rawtext)-(self.bracketing[after][0]-1)))

        return beforeindex, afterindex

    # the set of built-in identifiers which are also keywords,
    # i.e. keyword.iskeyword() returns True for them
    _ID_KEYWORDS = frozenset({"True", "False", "None"})

    @classmethod
    def _eat_identifier(cls, str, limit, pos):
        """Given a string and pos, return the number of chars in the
        identifier which ends at pos, or 0 if there is no such one.

        This ignores non-identifier eywords are not identifiers.
        """
        is_ascii_id_char = _IS_ASCII_ID_CHAR

        # Start at the end (pos) and work backwards.
        i = pos

        # Go backwards as long as the characters are valid ASCII
        # identifier characters. This is an optimization, since it
        # is faster in the common case where most of the characters
        # are ASCII.
        while i > limit and (
                ord(str[i - 1]) < 128 and
                is_ascii_id_char[ord(str[i - 1])]
        ):
            i -= 1

        # If the above loop ended due to reaching a non-ASCII
        # character, continue going backwards using the most generic
        # test for whether a string contains only valid identifier
        # characters.
        if i > limit and ord(str[i - 1]) >= 128:
            while i - 4 >= limit and ('a' + str[i - 4:pos]).isidentifier():
                i -= 4
            if i - 2 >= limit and ('a' + str[i - 2:pos]).isidentifier():
                i -= 2
            if i - 1 >= limit and ('a' + str[i - 1:pos]).isidentifier():
                i -= 1

            # The identifier candidate starts here. If it isn't a valid
            # identifier, don't eat anything. At this point that is only
            # possible if the first character isn't a valid first
            # character for an identifier.
            if not str[i:pos].isidentifier():
                return 0
        elif i < pos:
            # All characters in str[i:pos] are valid ASCII identifier
            # characters, so it is enough to check that the first is
            # valid as the first character of an identifier.
            if not _IS_ASCII_ID_FIRST_CHAR[ord(str[i])]:
                return 0

        # All keywords are valid identifiers, but should not be
        # considered identifiers here, except for True, False and None.
        if i < pos and (
                iskeyword(str[i:pos]) and
                str[i:pos] not in cls._ID_KEYWORDS
        ):
            return 0

        return pos - i

    # This string includes all chars that may be in a white space
    _whitespace_chars = " \t\n\\"

    def get_expression(self):
        """Return a string with the Python expression which ends at the
        given index, which is empty if there is no real one.
        """
        if not self.is_in_code():
            raise ValueError("get_expression should only be called "
                             "if index is inside a code.")

        rawtext = self.rawtext
        bracketing = self.bracketing

        brck_index = self.indexbracket
        brck_limit = bracketing[brck_index][0]
        pos = self.indexinrawtext

        last_identifier_pos = pos
        postdot_phase = True

        while True:
            # Eat whitespaces, comments, and if postdot_phase is False - a dot
            while True:
                if pos>brck_limit and rawtext[pos-1] in self._whitespace_chars:
                    # Eat a whitespace
                    pos -= 1
                elif (not postdot_phase and
                      pos > brck_limit and rawtext[pos-1] == '.'):
                    # Eat a dot
                    pos -= 1
                    postdot_phase = True
                # The next line will fail if we are *inside* a comment,
                # but we shouldn't be.
                elif (pos == brck_limit and brck_index > 0 and
                      rawtext[bracketing[brck_index-1][0]] == '#'):
                    # Eat a comment
                    brck_index -= 2
                    brck_limit = bracketing[brck_index][0]
                    pos = bracketing[brck_index+1][0]
                else:
                    # If we didn't eat anything, quit.
                    break

            if not postdot_phase:
                # We didn't find a dot, so the expression end at the
                # last identifier pos.
                break

            ret = self._eat_identifier(rawtext, brck_limit, pos)
            if ret:
                # There is an identifier to eat
                pos = pos - ret
                last_identifier_pos = pos
                # Now, to continue the search, we must find a dot.
                postdot_phase = False
                # (the loop continues now)

            elif pos == brck_limit:
                # We are at a bracketing limit. If it is a closing
                # bracket, eat the bracket, otherwise, stop the search.
                level = bracketing[brck_index][1]
                while brck_index > 0 and bracketing[brck_index-1][1] > level:
                    brck_index -= 1
                if bracketing[brck_index][0] == brck_limit:
                    # We were not at the end of a closing bracket
                    break
                pos = bracketing[brck_index][0]
                brck_index -= 1
                brck_limit = bracketing[brck_index][0]
                last_identifier_pos = pos
                if rawtext[pos] in "([":
                    # [] and () may be used after an identifier, so we
                    # continue. postdot_phase is True, so we don't allow a dot.
                    pass
                else:
                    # We can't continue after other types of brackets
                    if rawtext[pos] in "'\"":
                        # Scan a string prefix
                        while pos > 0 and rawtext[pos - 1] in "rRbBuU":
                            pos -= 1
                        last_identifier_pos = pos
                    break

            else:
                # We've found an operator or something.
                break

        return rawtext[last_identifier_pos:self.indexinrawtext]


if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_hyperparser', verbosity=2)
