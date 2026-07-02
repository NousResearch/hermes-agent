"""Extract, format and print information about Python stack traces."""

import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress

__all__ = ['extract_stack', 'extract_tb', 'format_exception',
           'format_exception_only', 'format_list', 'format_stack',
           'format_tb', 'print_exc', 'format_exc', 'print_exception',
           'print_last', 'print_stack', 'print_tb', 'clear_frames',
           'FrameSummary', 'StackSummary', 'TracebackException',
           'walk_stack', 'walk_tb']

#
# Formatting and printing lists of traceback lines.
#

def print_list(extracted_list, file=None):
    """Print the list of tuples as returned by extract_tb() or
    extract_stack() as a formatted stack trace to the given file."""
    if file is None:
        file = sys.stderr
    for item in StackSummary.from_list(extracted_list).format():
        print(item, file=file, end="")

def format_list(extracted_list):
    """Format a list of tuples or FrameSummary objects for printing.

    Given a list of tuples or FrameSummary objects as returned by
    extract_tb() or extract_stack(), return a list of strings ready
    for printing.

    Each string in the resulting list corresponds to the item with the
    same index in the argument list.  Each string ends in a newline;
    the strings may contain internal newlines as well, for those items
    whose source text line is not None.
    """
    return StackSummary.from_list(extracted_list).format()

#
# Printing and Extracting Tracebacks.
#

def print_tb(tb, limit=None, file=None):
    """Print up to 'limit' stack trace entries from the traceback 'tb'.

    If 'limit' is omitted or None, all entries are printed.  If 'file'
    is omitted or None, the output goes to sys.stderr; otherwise
    'file' should be an open file or file-like object with a write()
    method.
    """
    print_list(extract_tb(tb, limit=limit), file=file)

def format_tb(tb, limit=None):
    """A shorthand for 'format_list(extract_tb(tb, limit))'."""
    return extract_tb(tb, limit=limit).format()

def extract_tb(tb, limit=None):
    """
    Return a StackSummary object representing a list of
    pre-processed entries from traceback.

    This is useful for alternate formatting of stack traces.  If
    'limit' is omitted or None, all entries are extracted.  A
    pre-processed stack trace entry is a FrameSummary object
    containing attributes filename, lineno, name, and line
    representing the information that is usually printed for a stack
    trace.  The line is a string with leading and trailing
    whitespace stripped; if the source is not available it is None.
    """
    return StackSummary._extract_from_extended_frame_gen(
        _walk_tb_with_full_positions(tb), limit=limit)

#
# Exception formatting and output.
#

_cause_message = (
    "\nThe above exception was the direct cause "
    "of the following exception:\n\n")

_context_message = (
    "\nDuring handling of the above exception, "
    "another exception occurred:\n\n")


class _Sentinel:
    def __repr__(self):
        return "<implicit>"

_sentinel = _Sentinel()

def _parse_value_tb(exc, value, tb):
    if (value is _sentinel) != (tb is _sentinel):
        raise ValueError("Both or neither of value and tb must be given")
    if value is tb is _sentinel:
        if exc is not None:
            if isinstance(exc, BaseException):
                return exc, exc.__traceback__

            raise TypeError(f'Exception expected for value, '
                            f'{type(exc).__name__} found')
        else:
            return None, None
    return value, tb


def print_exception(exc, /, value=_sentinel, tb=_sentinel, limit=None, \
                    file=None, chain=True):
    """Print exception up to 'limit' stack trace entries from 'tb' to 'file'.

    This differs from print_tb() in the following ways: (1) if
    traceback is not None, it prints a header "Traceback (most recent
    call last):"; (2) it prints the exception type and value after the
    stack trace; (3) if type is SyntaxError and value has the
    appropriate format, it prints the line where the syntax error
    occurred with a caret on the next line indicating the approximate
    position of the error.
    """
    value, tb = _parse_value_tb(exc, value, tb)
    te = TracebackException(type(value), value, tb, limit=limit, compact=True)
    te.print(file=file, chain=chain)


def format_exception(exc, /, value=_sentinel, tb=_sentinel, limit=None, \
                     chain=True):
    """Format a stack trace and the exception information.

    The arguments have the same meaning as the corresponding arguments
    to print_exception().  The return value is a list of strings, each
    ending in a newline and some containing internal newlines.  When
    these lines are concatenated and printed, exactly the same text is
    printed as does print_exception().
    """
    value, tb = _parse_value_tb(exc, value, tb)
    te = TracebackException(type(value), value, tb, limit=limit, compact=True)
    return list(te.format(chain=chain))


def format_exception_only(exc, /, value=_sentinel):
    """Format the exception part of a traceback.

    The return value is a list of strings, each ending in a newline.

    The list contains the exception's message, which is
    normally a single string; however, for :exc:`SyntaxError` exceptions, it
    contains several lines that (when printed) display detailed information
    about where the syntax error occurred. Following the message, the list
    contains the exception's ``__notes__``.
    """
    if value is _sentinel:
        value = exc
    te = TracebackException(type(value), value, None, compact=True)
    return list(te.format_exception_only())


# -- not official API but folk probably use these two functions.

def _format_final_exc_line(etype, value):
    valuestr = _safe_string(value, 'exception')
    if value is None or not valuestr:
        line = "%s\n" % etype
    else:
        line = "%s: %s\n" % (etype, valuestr)
    return line

def _safe_string(value, what, func=str):
    try:
        return func(value)
    except:
        return f'<{what} {func.__name__}() failed>'

# --

def print_exc(limit=None, file=None, chain=True):
    """Shorthand for 'print_exception(*sys.exc_info(), limit, file)'."""
    print_exception(*sys.exc_info(), limit=limit, file=file, chain=chain)

def format_exc(limit=None, chain=True):
    """Like print_exc() but return a string."""
    return "".join(format_exception(*sys.exc_info(), limit=limit, chain=chain))

def print_last(limit=None, file=None, chain=True):
    """This is a shorthand for 'print_exception(sys.last_type,
    sys.last_value, sys.last_traceback, limit, file)'."""
    if not hasattr(sys, "last_type"):
        raise ValueError("no last exception")
    print_exception(sys.last_type, sys.last_value, sys.last_traceback,
                    limit, file, chain)

#
# Printing and Extracting Stacks.
#

def print_stack(f=None, limit=None, file=None):
    """Print a stack trace from its invocation point.

    The optional 'f' argument can be used to specify an alternate
    stack frame at which to start. The optional 'limit' and 'file'
    arguments have the same meaning as for print_exception().
    """
    if f is None:
        f = sys._getframe().f_back
    print_list(extract_stack(f, limit=limit), file=file)


def format_stack(f=None, limit=None):
    """Shorthand for 'format_list(extract_stack(f, limit))'."""
    if f is None:
        f = sys._getframe().f_back
    return format_list(extract_stack(f, limit=limit))


def extract_stack(f=None, limit=None):
    """Extract the raw traceback from the current stack frame.

    The return value has the same format as for extract_tb().  The
    optional 'f' and 'limit' arguments have the same meaning as for
    print_stack().  Each item in the list is a quadruple (filename,
    line number, function name, text), and the entries are in order
    from oldest to newest stack frame.
    """
    if f is None:
        f = sys._getframe().f_back
    stack = StackSummary.extract(walk_stack(f), limit=limit)
    stack.reverse()
    return stack


def clear_frames(tb):
    "Clear all references to local variables in the frames of a traceback."
    while tb is not None:
        try:
            tb.tb_frame.clear()
        except RuntimeError:
            # Ignore the exception raised if the frame is still executing.
            pass
        tb = tb.tb_next


class FrameSummary:
    """Information about a single frame from a traceback.

    - :attr:`filename` The filename for the frame.
    - :attr:`lineno` The line within filename for the frame that was
      active when the frame was captured.
    - :attr:`name` The name of the function or method that was executing
      when the frame was captured.
    - :attr:`line` The text from the linecache module for the
      of code that was running when the frame was captured.
    - :attr:`locals` Either None if locals were not supplied, or a dict
      mapping the name to the repr() of the variable.
    """

    __slots__ = ('filename', 'lineno', 'end_lineno', 'colno', 'end_colno',
                 'name', '_line', 'locals')

    def __init__(self, filename, lineno, name, *, lookup_line=True,
            locals=None, line=None,
            end_lineno=None, colno=None, end_colno=None):
        """Construct a FrameSummary.

        :param lookup_line: If True, `linecache` is consulted for the source
            code line. Otherwise, the line will be looked up when first needed.
        :param locals: If supplied the frame locals, which will be captured as
            object representations.
        :param line: If provided, use this instead of looking up the line in
            the linecache.
        """
        self.filename = filename
        self.lineno = lineno
        self.name = name
        self._line = line
        if lookup_line:
            self.line
        self.locals = {k: repr(v) for k, v in locals.items()} if locals else None
        self.end_lineno = end_lineno
        self.colno = colno
        self.end_colno = end_colno

    def __eq__(self, other):
        if isinstance(other, FrameSummary):
            return (self.filename == other.filename and
                    self.lineno == other.lineno and
                    self.name == other.name and
                    self.locals == other.locals)
        if isinstance(other, tuple):
            return (self.filename, self.lineno, self.name, self.line) == other
        return NotImplemented

    def __getitem__(self, pos):
        return (self.filename, self.lineno, self.name, self.line)[pos]

    def __iter__(self):
        return iter([self.filename, self.lineno, self.name, self.line])

    def __repr__(self):
        return "<FrameSummary file {filename}, line {lineno} in {name}>".format(
            filename=self.filename, lineno=self.lineno, name=self.name)

    def __len__(self):
        return 4

    @property
    def _original_line(self):
        # Returns the line as-is from the source, without modifying whitespace.
        self.line
        return self._line

    @property
    def line(self):
        if self._line is None:
            if self.lineno is None:
                return None
            self._line = linecache.getline(self.filename, self.lineno)
        return self._line.strip()


def walk_stack(f):
    """Walk a stack yielding the frame and line number for each frame.

    This will follow f.f_back from the given frame. If no frame is given, the
    current stack is used. Usually used with StackSummary.extract.
    """
    if f is None:
        f = sys._getframe().f_back.f_back.f_back.f_back
    while f is not None:
        yield f, f.f_lineno
        f = f.f_back


def walk_tb(tb):
    """Walk a traceback yielding the frame and line number for each frame.

    This will follow tb.tb_next (and thus is in the opposite order to
    walk_stack). Usually used with StackSummary.extract.
    """
    while tb is not None:
        yield tb.tb_frame, tb.tb_lineno
        tb = tb.tb_next


def _walk_tb_with_full_positions(tb):
    # Internal version of walk_tb that yields full code positions including
    # end line and column information.
    while tb is not None:
        positions = _get_code_position(tb.tb_frame.f_code, tb.tb_lasti)
        # Yield tb_lineno when co_positions does not have a line number to
        # maintain behavior with walk_tb.
        if positions[0] is None:
            yield tb.tb_frame, (tb.tb_lineno, ) + positions[1:]
        else:
            yield tb.tb_frame, positions
        tb = tb.tb_next


def _get_code_position(code, instruction_index):
    if instruction_index < 0:
        return (None, None, None, None)
    positions_gen = code.co_positions()
    return next(itertools.islice(positions_gen, instruction_index // 2, None))


_RECURSIVE_CUTOFF = 3 # Also hardcoded in traceback.c.

class StackSummary(list):
    """A list of FrameSummary objects, representing a stack of frames."""

    @classmethod
    def extract(klass, frame_gen, *, limit=None, lookup_lines=True,
            capture_locals=False):
        """Create a StackSummary from a traceback or stack object.

        :param frame_gen: A generator that yields (frame, lineno) tuples
            whose summaries are to be included in the stack.
        :param limit: None to include all frames or the number of frames to
            include.
        :param lookup_lines: If True, lookup lines for each frame immediately,
            otherwise lookup is deferred until the frame is rendered.
        :param capture_locals: If True, the local variables from each frame will
            be captured as object representations into the FrameSummary.
        """
        def extended_frame_gen():
            for f, lineno in frame_gen:
                yield f, (lineno, None, None, None)

        return klass._extract_from_extended_frame_gen(
            extended_frame_gen(), limit=limit, lookup_lines=lookup_lines,
            capture_locals=capture_locals)

    @classmethod
    def _extract_from_extended_frame_gen(klass, frame_gen, *, limit=None,
            lookup_lines=True, capture_locals=False):
        # Same as extract but operates on a frame generator that yields
        # (frame, (lineno, end_lineno, colno, end_colno)) in the stack.
        # Only lineno is required, the remaining fields can be None if the
        # information is not available.
        if limit is None:
            limit = getattr(sys, 'tracebacklimit', None)
            if limit is not None and limit < 0:
                limit = 0
        if limit is not None:
            if limit >= 0:
                frame_gen = itertools.islice(frame_gen, limit)
            else:
                frame_gen = collections.deque(frame_gen, maxlen=-limit)

        result = klass()
        fnames = set()
        for f, (lineno, end_lineno, colno, end_colno) in frame_gen:
            co = f.f_code
            filename = co.co_filename
            name = co.co_name

            fnames.add(filename)
            linecache.lazycache(filename, f.f_globals)
            # Must defer line lookups until we have called checkcache.
            if capture_locals:
                f_locals = f.f_locals
            else:
                f_locals = None
            result.append(FrameSummary(
                filename, lineno, name, lookup_line=False, locals=f_locals,
                end_lineno=end_lineno, colno=colno, end_colno=end_colno))
        for filename in fnames:
            linecache.checkcache(filename)
        # If immediate lookup was desired, trigger lookups now.
        if lookup_lines:
            for f in result:
                f.line
        return result

    @classmethod
    def from_list(klass, a_list):
        """
        Create a StackSummary object from a supplied list of
        FrameSummary objects or old-style list of tuples.
        """
        # While doing a fast-path check for isinstance(a_list, StackSummary) is
        # appealing, idlelib.run.cleanup_traceback and other similar code may
        # break this by making arbitrary frames plain tuples, so we need to
        # check on a frame by frame basis.
        result = StackSummary()
        for frame in a_list:
            if isinstance(frame, FrameSummary):
                result.append(frame)
            else:
                filename, lineno, name, line = frame
                result.append(FrameSummary(filename, lineno, name, line=line))
        return result

    def format_frame_summary(self, frame_summary):
        """Format the lines for a single FrameSummary.

        Returns a string representing one frame involved in the stack. This
        gets called for every frame to be printed in the stack summary.
        """
        row = []
        row.append('  File "{}", line {}, in {}\n'.format(
            frame_summary.filename, frame_summary.lineno, frame_summary.name))
        if frame_summary.line:
            stripped_line = frame_summary.line.strip()
            row.append('    {}\n'.format(stripped_line))

            line = frame_summary._original_line
            orig_line_len = len(line)
            frame_line_len = len(frame_summary.line.lstrip())
            stripped_characters = orig_line_len - frame_line_len
            if (
                frame_summary.colno is not None
                and frame_summary.end_colno is not None
            ):
                start_offset = _byte_offset_to_character_offset(
                    line, frame_summary.colno)
                end_offset = _byte_offset_to_character_offset(
                    line, frame_summary.end_colno)
                code_segment = line[start_offset:end_offset]

                anchors = None
                if frame_summary.lineno == frame_summary.end_lineno:
                    with suppress(Exception):
                        anchors = _extract_caret_anchors_from_line_segment(code_segment)
                else:
                    # Don't count the newline since the anchors only need to
                    # go up until the last character of the line.
                    end_offset = len(line.rstrip())

                # show indicators if primary char doesn't span the frame line
                if end_offset - start_offset < len(stripped_line) or (
                        anchors and anchors.right_start_offset - anchors.left_end_offset > 0):
                    # When showing this on a terminal, some of the non-ASCII characters
                    # might be rendered as double-width characters, so we need to take
                    # that into account when calculating the length of the line.
                    dp_start_offset = _display_width(line, start_offset) + 1
                    dp_end_offset = _display_width(line, end_offset) + 1

                    row.append('    ')
                    row.append(' ' * (dp_start_offset - stripped_characters))

                    if anchors:
                        dp_left_end_offset = _display_width(code_segment, anchors.left_end_offset)
                        dp_right_start_offset = _display_width(code_segment, anchors.right_start_offset)
                        row.append(anchors.primary_char * dp_left_end_offset)
                        row.append(anchors.secondary_char * (dp_right_start_offset - dp_left_end_offset))
                        row.append(anchors.primary_char * (dp_end_offset - dp_start_offset - dp_right_start_offset))
                    else:
                        row.append('^' * (dp_end_offset - dp_start_offset))

                    row.append('\n')

        if frame_summary.locals:
            for name, value in sorted(frame_summary.locals.items()):
                row.append('    {name} = {value}\n'.format(name=name, value=value))

        return ''.join(row)

    def format(self):
        """Format the stack ready for printing.

        Returns a list of strings ready for printing.  Each string in the
        resulting list corresponds to a single frame from the stack.
        Each string ends in a newline; the strings may contain internal
        newlines as well, for those items with source text lines.

        For long sequences of the same frame and line, the first few
        repetitions are shown, followed by a summary line stating the exact
        number of further repetitions.
        """
        result = []
        last_file = None
        last_line = None
        last_name = None
        count = 0
        for frame_summary in self:
            formatted_frame = self.format_frame_summary(frame_summary)
            if formatted_frame is None:
                continue
            if (last_file is None or last_file != frame_summary.filename or
                last_line is None or last_line != frame_summary.lineno or
                last_name is None or last_name != frame_summary.name):
                if count > _RECURSIVE_CUTOFF:
                    count -= _RECURSIVE_CUTOFF
                    result.append(
                        f'  [Previous line repeated {count} more '
                        f'time{"s" if count > 1 else ""}]\n'
                    )
                last_file = frame_summary.filename
                last_line = frame_summary.lineno
                last_name = frame_summary.name
                count = 0
            count += 1
            if count > _RECURSIVE_CUTOFF:
                continue
            result.append(formatted_frame)

        if count > _RECURSIVE_CUTOFF:
            count -= _RECURSIVE_CUTOFF
            result.append(
                f'  [Previous line repeated {count} more '
                f'time{"s" if count > 1 else ""}]\n'
            )
        return result


def _byte_offset_to_character_offset(str, offset):
    as_utf8 = str.encode('utf-8')
    return len(as_utf8[:offset].decode("utf-8", errors="replace"))


_Anchors = collections.namedtuple(
    "_Anchors",
    [
        "left_end_offset",
        "right_start_offset",
        "primary_char",
        "secondary_char",
    ],
    defaults=["~", "^"]
)

def _extract_caret_anchors_from_line_segment(segment):
    import ast

    try:
        tree = ast.parse(segment)
    except SyntaxError:
        return None

    if len(tree.body) != 1:
        return None

    normalize = lambda offset: _byte_offset_to_character_offset(segment, offset)
    statement = tree.body[0]
    match statement:
        case ast.Expr(expr):
            match expr:
                case ast.BinOp():
                    operator_start = normalize(expr.left.end_col_offset)
                    operator_end = normalize(expr.right.col_offset)
                    operator_str = segment[operator_start:operator_end]
                    operator_offset = len(operator_str) - len(operator_str.lstrip())

                    left_anchor = expr.left.end_col_offset + operator_offset
                    right_anchor = left_anchor + 1
                    if (
                        operator_offset + 1 < len(operator_str)
                        and not operator_str[operator_offset + 1].isspace()
                    ):
                        right_anchor += 1

                    while left_anchor < len(segment) and ((ch := segment[left_anchor]).isspace() or ch in ")#"):
                        left_anchor += 1
                        right_anchor += 1
                    return _Anchors(normalize(left_anchor), normalize(right_anchor))
                case ast.Subscript():
                    left_anchor = normalize(expr.value.end_col_offset)
                    right_anchor = normalize(expr.slice.end_col_offset + 1)
                    while left_anchor < len(segment) and ((ch := segment[left_anchor]).isspace() or ch != "["):
                        left_anchor += 1
                    while right_anchor < len(segment) and ((ch := segment[right_anchor]).isspace() or ch != "]"):
                        right_anchor += 1
                    if right_anchor < len(segment):
                        right_anchor += 1
                    return _Anchors(left_anchor, right_anchor)

    return None

_WIDE_CHAR_SPECIFIERS = "WF"

def _display_width(line, offset):
    """Calculate the extra amount of width space the given source
    code segment might take if it were to be displayed on a fixed
    width output device. Supports wide unicode characters and emojis."""

    # Fast track for ASCII-only strings
    if line.isascii():
        return offset

    import unicodedata

    return sum(
        2 if unicodedata.east_asian_width(char) in _WIDE_CHAR_SPECIFIERS else 1
        for char in line[:offset]
    )



class _ExceptionPrintContext:
    def __init__(self):
        self.seen = set()
        self.exception_group_depth = 0
        self.need_close = False

    def indent(self):
        return ' ' * (2 * self.exception_group_depth)

    def emit(self, text_gen, margin_char=None):
        if margin_char is None:
            margin_char = '|'
        indent_str = self.indent()
        if self.exception_group_depth:
            indent_str += margin_char + ' '

        if isinstance(text_gen, str):
            yield textwrap.indent(text_gen, indent_str, lambda line: True)
        else:
            for text in text_gen:
                yield textwrap.indent(text, indent_str, lambda line: True)


class TracebackException:
    """An exception ready for rendering.

    The traceback module captures enough attributes from the original exception
    to this intermediary form to ensure that no references are held, while
    still being able to fully print or format it.

    max_group_width and max_group_depth control the formatting of exception
    groups. The depth refers to the nesting level of the group, and the width
    refers to the size of a single exception group's exceptions array. The
    formatted output is truncated when either limit is exceeded.

    Use `from_exception` to create TracebackException instances from exception
    objects, or the constructor to create TracebackException instances from
    individual components.

    - :attr:`__cause__` A TracebackException of the original *__cause__*.
    - :attr:`__context__` A TracebackException of the original *__context__*.
    - :attr:`exceptions` For exception groups - a list of TracebackException
      instances for the nested *exceptions*.  ``None`` for other exceptions.
    - :attr:`__suppress_context__` The *__suppress_context__* value from the
      original exception.
    - :attr:`stack` A `StackSummary` representing the traceback.
    - :attr:`exc_type` The class of the original traceback.
    - :attr:`filename` For syntax errors - the filename where the error
      occurred.
    - :attr:`lineno` For syntax errors - the linenumber where the error
      occurred.
    - :attr:`end_lineno` For syntax errors - the end linenumber where the error
      occurred. Can be `None` if not present.
    - :attr:`text` For syntax errors - the text where the error
      occurred.
    - :attr:`offset` For syntax errors - the offset into the text where the
      error occurred.
    - :attr:`end_offset` For syntax errors - the end offset into the text where
      the error occurred. Can be `None` if not present.
    - :attr:`msg` For syntax errors - the compiler error message.
    """

    def __init__(self, exc_type, exc_value, exc_traceback, *, limit=None,
            lookup_lines=True, capture_locals=False, compact=False,
            max_group_width=15, max_group_depth=10, _seen=None):
        # NB: we need to accept exc_traceback, exc_value, exc_traceback to
        # permit backwards compat with the existing API, otherwise we
        # need stub thunk objects just to glue it together.
        # Handle loops in __cause__ or __context__.
        is_recursive_call = _seen is not None
        if _seen is None:
            _seen = set()
        _seen.add(id(exc_value))

        self.max_group_width = max_group_width
        self.max_group_depth = max_group_depth

        self.stack = StackSummary._extract_from_extended_frame_gen(
            _walk_tb_with_full_positions(exc_traceback),
            limit=limit, lookup_lines=lookup_lines,
            capture_locals=capture_locals)
        self.exc_type = exc_type
        # Capture now to permit freeing resources: only complication is in the
        # unofficial API _format_final_exc_line
        self._str = _safe_string(exc_value, 'exception')
        try:
            self.__notes__ = getattr(exc_value, '__notes__', None)
        except Exception as e:
            self.__notes__ = [
                f'Ignored error getting __notes__: {_safe_string(e, "__notes__", repr)}']

        if exc_type and issubclass(exc_type, SyntaxError):
            # Handle SyntaxError's specially
            self.filename = exc_value.filename
            lno = exc_value.lineno
            self.lineno = str(lno) if lno is not None else None
            end_lno = exc_value.end_lineno
            self.end_lineno = str(end_lno) if end_lno is not None else None
            self.text = exc_value.text
            self.offset = exc_value.offset
            self.end_offset = exc_value.end_offset
            self.msg = exc_value.msg
        if lookup_lines:
            self._load_lines()
        self.__suppress_context__ = \
            exc_value.__suppress_context__ if exc_value is not None else False

        # Convert __cause__ and __context__ to `TracebackExceptions`s, use a
        # queue to avoid recursion (only the top-level call gets _seen == None)
        if not is_recursive_call:
            queue = [(self, exc_value)]
            while queue:
                te, e = queue.pop()
                if (e and e.__cause__ is not None
                    and id(e.__cause__) not in _seen):
                    cause = TracebackException(
                        type(e.__cause__),
                        e.__cause__,
                        e.__cause__.__traceback__,
                        limit=limit,
                        lookup_lines=lookup_lines,
                        capture_locals=capture_locals,
                        max_group_width=max_group_width,
                        max_group_depth=max_group_depth,
                        _seen=_seen)
                else:
                    cause = None

                if compact:
                    need_context = (cause is None and
                                    e is not None and
                                    not e.__suppress_context__)
                else:
                    need_context = True
                if (e and e.__context__ is not None
                    and need_context and id(e.__context__) not in _seen):
                    context = TracebackException(
                        type(e.__context__),
                        e.__context__,
                        e.__context__.__traceback__,
                        limit=limit,
                        lookup_lines=lookup_lines,
                        capture_locals=capture_locals,
                        max_group_width=max_group_width,
                        max_group_depth=max_group_depth,
                        _seen=_seen)
                else:
                    context = None

                if e and isinstance(e, BaseExceptionGroup):
                    exceptions = []
                    for exc in e.exceptions:
                        texc = TracebackException(
                            type(exc),
                            exc,
                            exc.__traceback__,
                            limit=limit,
                            lookup_lines=lookup_lines,
                            capture_locals=capture_locals,
                            max_group_width=max_group_width,
                            max_group_depth=max_group_depth,
                            _seen=_seen)
                        exceptions.append(texc)
                else:
                    exceptions = None

                te.__cause__ = cause
                te.__context__ = context
                te.exceptions = exceptions
                if cause:
                    queue.append((te.__cause__, e.__cause__))
                if context:
                    queue.append((te.__context__, e.__context__))
                if exceptions:
                    queue.extend(zip(te.exceptions, e.exceptions))

    @classmethod
    def from_exception(cls, exc, *args, **kwargs):
        """Create a TracebackException from an exception."""
        return cls(type(exc), exc, exc.__traceback__, *args, **kwargs)

    def _load_lines(self):
        """Private API. force all lines in the stack to be loaded."""
        for frame in self.stack:
            frame.line

    def __eq__(self, other):
        if isinstance(other, TracebackException):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __str__(self):
        return self._str

    def format_exception_only(self):
        """Format the exception part of the traceback.

        The return value is a generator of strings, each ending in a newline.

        Generator yields the exception message.
        For :exc:`SyntaxError` exceptions, it
        also yields (before the exception message)
        several lines that (when printed)
        display detailed information about where the syntax error occurred.
        Following the message, generator also yields
        all the exception's ``__notes__``.
        """
        if self.exc_type is None:
            yield _format_final_exc_line(None, self._str)
            return

        stype = self.exc_type.__qualname__
        smod = self.exc_type.__module__
        if smod not in ("__main__", "builtins"):
            if not isinstance(smod, str):
                smod = "<unknown>"
            stype = smod + '.' + stype

        if not issubclass(self.exc_type, SyntaxError):
            yield _format_final_exc_line(stype, self._str)
        else:
            yield from self._format_syntax_error(stype)
        if isinstance(self.__notes__, collections.abc.Sequence):
            for note in self.__notes__:
                note = _safe_string(note, 'note')
                yield from [l + '\n' for l in note.split('\n')]
        elif self.__notes__ is not None:
            yield _safe_string(self.__notes__, '__notes__', func=repr)

    def _format_syntax_error(self, stype):
        """Format SyntaxError exceptions (internal helper)."""
        # Show exactly where the problem was found.
        filename_suffix = ''
        if self.lineno is not None:
            yield '  File "{}", line {}\n'.format(
                self.filename or "<string>", self.lineno)
        elif self.filename is not None:
            filename_suffix = ' ({})'.format(self.filename)

        text = self.text
        if text is not None:
            # text  = "   foo\n"
            # rtext = "   foo"
            # ltext =    "foo"
            rtext = text.rstrip('\n')
            ltext = rtext.lstrip(' \n\f')
            spaces = len(rtext) - len(ltext)
            yield '    {}\n'.format(ltext)

            if self.offset is not None:
                offset = self.offset
                end_offset = self.end_offset if self.end_offset not in {None, 0} else offset
                if offset == end_offset or end_offset == -1:
                    end_offset = offset + 1

                # Convert 1-based column offset to 0-based index into stripped text
                colno = offset - 1 - spaces
                end_colno = end_offset - 1 - spaces
                if colno >= 0:
                    # non-space whitespace (likes tabs) must be kept for alignment
                    caretspace = ((c if c.isspace() else ' ') for c in ltext[:colno])
                    yield '    {}{}'.format("".join(caretspace), ('^' * (end_colno - colno) + "\n"))
        msg = self.msg or "<no detail available>"
        yield "{}: {}{}\n".format(stype, msg, filename_suffix)

    def format(self, *, chain=True, _ctx=None):
        """Format the exception.

        If chain is not *True*, *__cause__* and *__context__* will not be formatted.

        The return value is a generator of strings, each ending in a newline and
        some containing internal newlines. `print_exception` is a wrapper around
        this method which just prints the lines to a file.

        The message indicating which exception occurred is always the last
        string in the output.
        """

        if _ctx is None:
            _ctx = _ExceptionPrintContext()

        output = []
        exc = self
        if chain:
            while exc:
                if exc.__cause__ is not None:
                    chained_msg = _cause_message
                    chained_exc = exc.__cause__
                elif (exc.__context__  is not None and
                      not exc.__suppress_context__):
                    chained_msg = _context_message
                    chained_exc = exc.__context__
                else:
                    chained_msg = None
                    chained_exc = None

                output.append((chained_msg, exc))
                exc = chained_exc
        else:
            output.append((None, exc))

        for msg, exc in reversed(output):
            if msg is not None:
                yield from _ctx.emit(msg)
            if exc.exceptions is None:
                if exc.stack:
                    yield from _ctx.emit('Traceback (most recent call last):\n')
                    yield from _ctx.emit(exc.stack.format())
                yield from _ctx.emit(exc.format_exception_only())
            elif _ctx.exception_group_depth > self.max_group_depth:
                # exception group, but depth exceeds limit
                yield from _ctx.emit(
                    f"... (max_group_depth is {self.max_group_depth})\n")
            else:
                # format exception group
                is_toplevel = (_ctx.exception_group_depth == 0)
                if is_toplevel:
                    _ctx.exception_group_depth += 1

                if exc.stack:
                    yield from _ctx.emit(
                        'Exception Group Traceback (most recent call last):\n',
                        margin_char = '+' if is_toplevel else None)
                    yield from _ctx.emit(exc.stack.format())

                yield from _ctx.emit(exc.format_exception_only())
                num_excs = len(exc.exceptions)
                if num_excs <= self.max_group_width:
                    n = num_excs
                else:
                    n = self.max_group_width + 1
                _ctx.need_close = False
                for i in range(n):
                    last_exc = (i == n-1)
                    if last_exc:
                        # The closing frame may be added by a recursive call
                        _ctx.need_close = True

                    if self.max_group_width is not None:
                        truncated = (i >= self.max_group_width)
                    else:
                        truncated = False
                    title = f'{i+1}' if not truncated else '...'
                    yield (_ctx.indent() +
                           ('+-' if i==0 else '  ') +
                           f'+---------------- {title} ----------------\n')
                    _ctx.exception_group_depth += 1
                    if not truncated:
                        yield from exc.exceptions[i].format(chain=chain, _ctx=_ctx)
                    else:
                        remaining = num_excs - self.max_group_width
                        plural = 's' if remaining > 1 else ''
                        yield from _ctx.emit(
                            f"and {remaining} more exception{plural}\n")

                    if last_exc and _ctx.need_close:
                        yield (_ctx.indent() +
                               "+------------------------------------\n")
                        _ctx.need_close = False
                    _ctx.exception_group_depth -= 1

                if is_toplevel:
                    assert _ctx.exception_group_depth == 1
                    _ctx.exception_group_depth = 0


    def print(self, *, file=None, chain=True):
        """Print the result of self.format(chain=chain) to 'file'."""
        if file is None:
            file = sys.stderr
        for line in self.format(chain=chain):
            print(line, file=file, end="")
