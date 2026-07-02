# Copyright 2004-2005 Elemental Security, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

# Modifications:
# Copyright 2006 Google, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Parser driver.

This provides a high-level interface to parse a file into a syntax tree.

"""

__author__ = "Guido van Rossum <guido@python.org>"

__all__ = ["Driver", "load_grammar"]

# Python imports
import io
import os
import logging
import pkgutil
import sys

# Pgen imports
from . import grammar, parse, token, tokenize, pgen


class Driver(object):

    def __init__(self, grammar, convert=None, logger=None):
        self.grammar = grammar
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        self.convert = convert

    def parse_tokens(self, tokens, debug=False):
        """Parse a series of tokens and return the syntax tree."""
        # XXX Move the prefix computation into a wrapper around tokenize.
        p = parse.Parser(self.grammar, self.convert)
        p.setup()
        lineno = 1
        column = 0
        type = value = start = end = line_text = None
        prefix = ""
        for quintuple in tokens:
            type, value, start, end, line_text = quintuple
            if start != (lineno, column):
                assert (lineno, column) <= start, ((lineno, column), start)
                s_lineno, s_column = start
                if lineno < s_lineno:
                    prefix += "\n" * (s_lineno - lineno)
                    lineno = s_lineno
                    column = 0
                if column < s_column:
                    prefix += line_text[column:s_column]
                    column = s_column
            if type in (tokenize.COMMENT, tokenize.NL):
                prefix += value
                lineno, column = end
                if value.endswith("\n"):
                    lineno += 1
                    column = 0
                continue
            if type == token.OP:
                type = grammar.opmap[value]
            if debug:
                self.logger.debug("%s %r (prefix=%r)",
                                  token.tok_name[type], value, prefix)
            if p.addtoken(type, value, (prefix, start)):
                if debug:
                    self.logger.debug("Stop.")
                break
            prefix = ""
            lineno, column = end
            if value.endswith("\n"):
                lineno += 1
                column = 0
        else:
            # We never broke out -- EOF is too soon (how can this happen???)
            raise parse.ParseError("incomplete input",
                                   type, value, (prefix, start))
        return p.rootnode

    def parse_stream_raw(self, stream, debug=False):
        """Parse a stream and return the syntax tree."""
        tokens = tokenize.generate_tokens(stream.readline)
        return self.parse_tokens(tokens, debug)

    def parse_stream(self, stream, debug=False):
        """Parse a stream and return the syntax tree."""
        return self.parse_stream_raw(stream, debug)

    def parse_file(self, filename, encoding=None, debug=False):
        """Parse a file and return the syntax tree."""
        with io.open(filename, "r", encoding=encoding) as stream:
            return self.parse_stream(stream, debug)

    def parse_string(self, text, debug=False):
        """Parse a string and return the syntax tree."""
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        return self.parse_tokens(tokens, debug)


def _generate_pickle_name(gt):
    head, tail = os.path.splitext(gt)
    if tail == ".txt":
        tail = ""
    return head + tail + ".".join(map(str, sys.version_info)) + ".pickle"


def load_grammar(gt="Grammar.txt", gp=None,
                 save=True, force=False, logger=None):
    """Load the grammar (maybe from a pickle)."""
    if logger is None:
        logger = logging.getLogger()
    gp = _generate_pickle_name(gt) if gp is None else gp
    if force or not _newer(gp, gt):
        logger.info("Generating grammar tables from %s", gt)
        g = pgen.generate_grammar(gt)
        if save:
            logger.info("Writing grammar tables to %s", gp)
            try:
                g.dump(gp)
            except OSError as e:
                logger.info("Writing failed: %s", e)
    else:
        g = grammar.Grammar()
        g.load(gp)
    return g


def _newer(a, b):
    """Inquire whether file a was written since file b."""
    if not os.path.exists(a):
        return False
    if not os.path.exists(b):
        return True
    return os.path.getmtime(a) >= os.path.getmtime(b)


def load_packaged_grammar(package, grammar_source):
    """Normally, loads a pickled grammar by doing
        pkgutil.get_data(package, pickled_grammar)
    where *pickled_grammar* is computed from *grammar_source* by adding the
    Python version and using a ``.pickle`` extension.

    However, if *grammar_source* is an extant file, load_grammar(grammar_source)
    is called instead. This facilitates using a packaged grammar file when needed
    but preserves load_grammar's automatic regeneration behavior when possible.

    """
    if os.path.isfile(grammar_source):
        return load_grammar(grammar_source)
    pickled_name = _generate_pickle_name(os.path.basename(grammar_source))
    data = pkgutil.get_data(package, pickled_name)
    g = grammar.Grammar()
    g.loads(data)
    return g


def main(*args):
    """Main program, when run as a script: produce grammar pickle files.

    Calls load_grammar for each argument, a path to a grammar text file.
    """
    if not args:
        args = sys.argv[1:]
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(message)s')
    for gt in args:
        load_grammar(gt, save=True, force=True)
    return True

if __name__ == "__main__":
    sys.exit(int(not main()))
