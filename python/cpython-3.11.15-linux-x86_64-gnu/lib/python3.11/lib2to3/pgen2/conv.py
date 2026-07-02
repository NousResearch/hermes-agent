# Copyright 2004-2005 Elemental Security, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Convert graminit.[ch] spit out by pgen to Python code.

Pgen is the Python parser generator.  It is useful to quickly create a
parser from a grammar file in Python's grammar notation.  But I don't
want my parsers to be written in C (yet), so I'm translating the
parsing tables to Python data structures and writing a Python parse
engine.

Note that the token numbers are constants determined by the standard
Python tokenizer.  The standard token module defines these numbers and
their names (the names are not used much).  The token numbers are
hardcoded into the Python tokenizer and into pgen.  A Python
implementation of the Python tokenizer is also available, in the
standard tokenize module.

On the other hand, symbol numbers (representing the grammar's
non-terminals) are assigned by pgen based on the actual grammar
input.

Note: this module is pretty much obsolete; the pgen module generates
equivalent grammar tables directly from the Grammar.txt input file
without having to invoke the Python pgen C program.

"""

# Python imports
import re

# Local imports
from pgen2 import grammar, token


class Converter(grammar.Grammar):
    """Grammar subclass that reads classic pgen output files.

    The run() method reads the tables as produced by the pgen parser
    generator, typically contained in two C files, graminit.h and
    graminit.c.  The other methods are for internal use only.

    See the base class for more documentation.

    """

    def run(self, graminit_h, graminit_c):
        """Load the grammar tables from the text files written by pgen."""
        self.parse_graminit_h(graminit_h)
        self.parse_graminit_c(graminit_c)
        self.finish_off()

    def parse_graminit_h(self, filename):
        """Parse the .h file written by pgen.  (Internal)

        This file is a sequence of #define statements defining the
        nonterminals of the grammar as numbers.  We build two tables
        mapping the numbers to names and back.

        """
        try:
            f = open(filename)
        except OSError as err:
            print("Can't open %s: %s" % (filename, err))
            return False
        self.symbol2number = {}
        self.number2symbol = {}
        lineno = 0
        for line in f:
            lineno += 1
            mo = re.match(r"^#define\s+(\w+)\s+(\d+)$", line)
            if not mo and line.strip():
                print("%s(%s): can't parse %s" % (filename, lineno,
                                                  line.strip()))
            else:
                symbol, number = mo.groups()
                number = int(number)
                assert symbol not in self.symbol2number
                assert number not in self.number2symbol
                self.symbol2number[symbol] = number
                self.number2symbol[number] = symbol
        return True

    def parse_graminit_c(self, filename):
        """Parse the .c file written by pgen.  (Internal)

        The file looks as follows.  The first two lines are always this:

        #include "pgenheaders.h"
        #include "grammar.h"

        After that come four blocks:

        1) one or more state definitions
        2) a table defining dfas
        3) a table defining labels
        4) a struct defining the grammar

        A state definition has the following form:
        - one or more arc arrays, each of the form:
          static arc arcs_<n>_<m>[<k>] = {
                  {<i>, <j>},
                  ...
          };
        - followed by a state array, of the form:
          static state states_<s>[<t>] = {
                  {<k>, arcs_<n>_<m>},
                  ...
          };

        """
        try:
            f = open(filename)
        except OSError as err:
            print("Can't open %s: %s" % (filename, err))
            return False
        # The code below essentially uses f's iterator-ness!
        lineno = 0

        # Expect the two #include lines
        lineno, line = lineno+1, next(f)
        assert line == '#include "pgenheaders.h"\n', (lineno, line)
        lineno, line = lineno+1, next(f)
        assert line == '#include "grammar.h"\n', (lineno, line)

        # Parse the state definitions
        lineno, line = lineno+1, next(f)
        allarcs = {}
        states = []
        while line.startswith("static arc "):
            while line.startswith("static arc "):
                mo = re.match(r"static arc arcs_(\d+)_(\d+)\[(\d+)\] = {$",
                              line)
                assert mo, (lineno, line)
                n, m, k = list(map(int, mo.groups()))
                arcs = []
                for _ in range(k):
                    lineno, line = lineno+1, next(f)
                    mo = re.match(r"\s+{(\d+), (\d+)},$", line)
                    assert mo, (lineno, line)
                    i, j = list(map(int, mo.groups()))
                    arcs.append((i, j))
                lineno, line = lineno+1, next(f)
                assert line == "};\n", (lineno, line)
                allarcs[(n, m)] = arcs
                lineno, line = lineno+1, next(f)
            mo = re.match(r"static state states_(\d+)\[(\d+)\] = {$", line)
            assert mo, (lineno, line)
            s, t = list(map(int, mo.groups()))
            assert s == len(states), (lineno, line)
            state = []
            for _ in range(t):
                lineno, line = lineno+1, next(f)
                mo = re.match(r"\s+{(\d+), arcs_(\d+)_(\d+)},$", line)
                assert mo, (lineno, line)
                k, n, m = list(map(int, mo.groups()))
                arcs = allarcs[n, m]
                assert k == len(arcs), (lineno, line)
                state.append(arcs)
            states.append(state)
            lineno, line = lineno+1, next(f)
            assert line == "};\n", (lineno, line)
            lineno, line = lineno+1, next(f)
        self.states = states

        # Parse the dfas
        dfas = {}
        mo = re.match(r"static dfa dfas\[(\d+)\] = {$", line)
        assert mo, (lineno, line)
        ndfas = int(mo.group(1))
        for i in range(ndfas):
            lineno, line = lineno+1, next(f)
            mo = re.match(r'\s+{(\d+), "(\w+)", (\d+), (\d+), states_(\d+),$',
                          line)
            assert mo, (lineno, line)
            symbol = mo.group(2)
            number, x, y, z = list(map(int, mo.group(1, 3, 4, 5)))
            assert self.symbol2number[symbol] == number, (lineno, line)
            assert self.number2symbol[number] == symbol, (lineno, line)
            assert x == 0, (lineno, line)
            state = states[z]
            assert y == len(state), (lineno, line)
            lineno, line = lineno+1, next(f)
            mo = re.match(r'\s+("(?:\\\d\d\d)*")},$', line)
            assert mo, (lineno, line)
            first = {}
            rawbitset = eval(mo.group(1))
            for i, c in enumerate(rawbitset):
                byte = ord(c)
                for j in range(8):
                    if byte & (1<<j):
                        first[i*8 + j] = 1
            dfas[number] = (state, first)
        lineno, line = lineno+1, next(f)
        assert line == "};\n", (lineno, line)
        self.dfas = dfas

        # Parse the labels
        labels = []
        lineno, line = lineno+1, next(f)
        mo = re.match(r"static label labels\[(\d+)\] = {$", line)
        assert mo, (lineno, line)
        nlabels = int(mo.group(1))
        for i in range(nlabels):
            lineno, line = lineno+1, next(f)
            mo = re.match(r'\s+{(\d+), (0|"\w+")},$', line)
            assert mo, (lineno, line)
            x, y = mo.groups()
            x = int(x)
            if y == "0":
                y = None
            else:
                y = eval(y)
            labels.append((x, y))
        lineno, line = lineno+1, next(f)
        assert line == "};\n", (lineno, line)
        self.labels = labels

        # Parse the grammar struct
        lineno, line = lineno+1, next(f)
        assert line == "grammar _PyParser_Grammar = {\n", (lineno, line)
        lineno, line = lineno+1, next(f)
        mo = re.match(r"\s+(\d+),$", line)
        assert mo, (lineno, line)
        ndfas = int(mo.group(1))
        assert ndfas == len(self.dfas)
        lineno, line = lineno+1, next(f)
        assert line == "\tdfas,\n", (lineno, line)
        lineno, line = lineno+1, next(f)
        mo = re.match(r"\s+{(\d+), labels},$", line)
        assert mo, (lineno, line)
        nlabels = int(mo.group(1))
        assert nlabels == len(self.labels), (lineno, line)
        lineno, line = lineno+1, next(f)
        mo = re.match(r"\s+(\d+)$", line)
        assert mo, (lineno, line)
        start = int(mo.group(1))
        assert start in self.number2symbol, (lineno, line)
        self.start = start
        lineno, line = lineno+1, next(f)
        assert line == "};\n", (lineno, line)
        try:
            lineno, line = lineno+1, next(f)
        except StopIteration:
            pass
        else:
            assert 0, (lineno, line)

    def finish_off(self):
        """Create additional useful structures.  (Internal)."""
        self.keywords = {} # map from keyword strings to arc labels
        self.tokens = {}   # map from numeric token values to arc labels
        for ilabel, (type, value) in enumerate(self.labels):
            if type == token.NAME and value is not None:
                self.keywords[value] = ilabel
            elif value is None:
                self.tokens[type] = ilabel
