# Copyright 2006 Google, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""
Python parse tree definitions.

This is a very concrete parse tree; we need to keep every token and
even the comments and whitespace between tokens.

There's also a pattern matching implementation here.
"""

__author__ = "Guido van Rossum <guido@python.org>"

import sys
from io import StringIO

HUGE = 0x7FFFFFFF  # maximum repeat count, default max

_type_reprs = {}
def type_repr(type_num):
    global _type_reprs
    if not _type_reprs:
        from .pygram import python_symbols
        # printing tokens is possible but not as useful
        # from .pgen2 import token // token.__dict__.items():
        for name, val in python_symbols.__dict__.items():
            if type(val) == int: _type_reprs[val] = name
    return _type_reprs.setdefault(type_num, type_num)

class Base(object):

    """
    Abstract base class for Node and Leaf.

    This provides some default functionality and boilerplate using the
    template pattern.

    A node may be a subnode of at most one parent.
    """

    # Default values for instance variables
    type = None    # int: token number (< 256) or symbol number (>= 256)
    parent = None  # Parent node pointer, or None
    children = ()  # Tuple of subnodes
    was_changed = False
    was_checked = False

    def __new__(cls, *args, **kwds):
        """Constructor that prevents Base from being instantiated."""
        assert cls is not Base, "Cannot instantiate Base"
        return object.__new__(cls)

    def __eq__(self, other):
        """
        Compare two nodes for equality.

        This calls the method _eq().
        """
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self._eq(other)

    __hash__ = None # For Py3 compatibility.

    def _eq(self, other):
        """
        Compare two nodes for equality.

        This is called by __eq__ and __ne__.  It is only called if the two nodes
        have the same type.  This must be implemented by the concrete subclass.
        Nodes should be considered equal if they have the same structure,
        ignoring the prefix string and other context information.
        """
        raise NotImplementedError

    def clone(self):
        """
        Return a cloned (deep) copy of self.

        This must be implemented by the concrete subclass.
        """
        raise NotImplementedError

    def post_order(self):
        """
        Return a post-order iterator for the tree.

        This must be implemented by the concrete subclass.
        """
        raise NotImplementedError

    def pre_order(self):
        """
        Return a pre-order iterator for the tree.

        This must be implemented by the concrete subclass.
        """
        raise NotImplementedError

    def replace(self, new):
        """Replace this node with a new one in the parent."""
        assert self.parent is not None, str(self)
        assert new is not None
        if not isinstance(new, list):
            new = [new]
        l_children = []
        found = False
        for ch in self.parent.children:
            if ch is self:
                assert not found, (self.parent.children, self, new)
                if new is not None:
                    l_children.extend(new)
                found = True
            else:
                l_children.append(ch)
        assert found, (self.children, self, new)
        self.parent.changed()
        self.parent.children = l_children
        for x in new:
            x.parent = self.parent
        self.parent = None

    def get_lineno(self):
        """Return the line number which generated the invocant node."""
        node = self
        while not isinstance(node, Leaf):
            if not node.children:
                return
            node = node.children[0]
        return node.lineno

    def changed(self):
        if self.parent:
            self.parent.changed()
        self.was_changed = True

    def remove(self):
        """
        Remove the node from the tree. Returns the position of the node in its
        parent's children before it was removed.
        """
        if self.parent:
            for i, node in enumerate(self.parent.children):
                if node is self:
                    self.parent.changed()
                    del self.parent.children[i]
                    self.parent = None
                    return i

    @property
    def next_sibling(self):
        """
        The node immediately following the invocant in their parent's children
        list. If the invocant does not have a next sibling, it is None
        """
        if self.parent is None:
            return None

        # Can't use index(); we need to test by identity
        for i, child in enumerate(self.parent.children):
            if child is self:
                try:
                    return self.parent.children[i+1]
                except IndexError:
                    return None

    @property
    def prev_sibling(self):
        """
        The node immediately preceding the invocant in their parent's children
        list. If the invocant does not have a previous sibling, it is None.
        """
        if self.parent is None:
            return None

        # Can't use index(); we need to test by identity
        for i, child in enumerate(self.parent.children):
            if child is self:
                if i == 0:
                    return None
                return self.parent.children[i-1]

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def depth(self):
        if self.parent is None:
            return 0
        return 1 + self.parent.depth()

    def get_suffix(self):
        """
        Return the string immediately following the invocant node. This is
        effectively equivalent to node.next_sibling.prefix
        """
        next_sib = self.next_sibling
        if next_sib is None:
            return ""
        return next_sib.prefix

    if sys.version_info < (3, 0):
        def __str__(self):
            return str(self).encode("ascii")

class Node(Base):

    """Concrete implementation for interior nodes."""

    def __init__(self,type, children,
                 context=None,
                 prefix=None,
                 fixers_applied=None):
        """
        Initializer.

        Takes a type constant (a symbol number >= 256), a sequence of
        child nodes, and an optional context keyword argument.

        As a side effect, the parent pointers of the children are updated.
        """
        assert type >= 256, type
        self.type = type
        self.children = list(children)
        for ch in self.children:
            assert ch.parent is None, repr(ch)
            ch.parent = self
        if prefix is not None:
            self.prefix = prefix
        if fixers_applied:
            self.fixers_applied = fixers_applied[:]
        else:
            self.fixers_applied = None

    def __repr__(self):
        """Return a canonical string representation."""
        return "%s(%s, %r)" % (self.__class__.__name__,
                               type_repr(self.type),
                               self.children)

    def __unicode__(self):
        """
        Return a pretty string representation.

        This reproduces the input source exactly.
        """
        return "".join(map(str, self.children))

    if sys.version_info > (3, 0):
        __str__ = __unicode__

    def _eq(self, other):
        """Compare two nodes for equality."""
        return (self.type, self.children) == (other.type, other.children)

    def clone(self):
        """Return a cloned (deep) copy of self."""
        return Node(self.type, [ch.clone() for ch in self.children],
                    fixers_applied=self.fixers_applied)

    def post_order(self):
        """Return a post-order iterator for the tree."""
        for child in self.children:
            yield from child.post_order()
        yield self

    def pre_order(self):
        """Return a pre-order iterator for the tree."""
        yield self
        for child in self.children:
            yield from child.pre_order()

    @property
    def prefix(self):
        """
        The whitespace and comments preceding this node in the input.
        """
        if not self.children:
            return ""
        return self.children[0].prefix

    @prefix.setter
    def prefix(self, prefix):
        if self.children:
            self.children[0].prefix = prefix

    def set_child(self, i, child):
        """
        Equivalent to 'node.children[i] = child'. This method also sets the
        child's parent attribute appropriately.
        """
        child.parent = self
        self.children[i].parent = None
        self.children[i] = child
        self.changed()

    def insert_child(self, i, child):
        """
        Equivalent to 'node.children.insert(i, child)'. This method also sets
        the child's parent attribute appropriately.
        """
        child.parent = self
        self.children.insert(i, child)
        self.changed()

    def append_child(self, child):
        """
        Equivalent to 'node.children.append(child)'. This method also sets the
        child's parent attribute appropriately.
        """
        child.parent = self
        self.children.append(child)
        self.changed()


class Leaf(Base):

    """Concrete implementation for leaf nodes."""

    # Default values for instance variables
    _prefix = ""  # Whitespace and comments preceding this token in the input
    lineno = 0    # Line where this token starts in the input
    column = 0    # Column where this token tarts in the input

    def __init__(self, type, value,
                 context=None,
                 prefix=None,
                 fixers_applied=[]):
        """
        Initializer.

        Takes a type constant (a token number < 256), a string value, and an
        optional context keyword argument.
        """
        assert 0 <= type < 256, type
        if context is not None:
            self._prefix, (self.lineno, self.column) = context
        self.type = type
        self.value = value
        if prefix is not None:
            self._prefix = prefix
        self.fixers_applied = fixers_applied[:]

    def __repr__(self):
        """Return a canonical string representation."""
        return "%s(%r, %r)" % (self.__class__.__name__,
                               self.type,
                               self.value)

    def __unicode__(self):
        """
        Return a pretty string representation.

        This reproduces the input source exactly.
        """
        return self.prefix + str(self.value)

    if sys.version_info > (3, 0):
        __str__ = __unicode__

    def _eq(self, other):
        """Compare two nodes for equality."""
        return (self.type, self.value) == (other.type, other.value)

    def clone(self):
        """Return a cloned (deep) copy of self."""
        return Leaf(self.type, self.value,
                    (self.prefix, (self.lineno, self.column)),
                    fixers_applied=self.fixers_applied)

    def leaves(self):
        yield self

    def post_order(self):
        """Return a post-order iterator for the tree."""
        yield self

    def pre_order(self):
        """Return a pre-order iterator for the tree."""
        yield self

    @property
    def prefix(self):
        """
        The whitespace and comments preceding this token in the input.
        """
        return self._prefix

    @prefix.setter
    def prefix(self, prefix):
        self.changed()
        self._prefix = prefix

def convert(gr, raw_node):
    """
    Convert raw node information to a Node or Leaf instance.

    This is passed to the parser driver which calls it whenever a reduction of a
    grammar rule produces a new complete node, so that the tree is build
    strictly bottom-up.
    """
    type, value, context, children = raw_node
    if children or type in gr.number2symbol:
        # If there's exactly one child, return that child instead of
        # creating a new node.
        if len(children) == 1:
            return children[0]
        return Node(type, children, context=context)
    else:
        return Leaf(type, value, context=context)


class BasePattern(object):

    """
    A pattern is a tree matching pattern.

    It looks for a specific node type (token or symbol), and
    optionally for a specific content.

    This is an abstract base class.  There are three concrete
    subclasses:

    - LeafPattern matches a single leaf node;
    - NodePattern matches a single node (usually non-leaf);
    - WildcardPattern matches a sequence of nodes of variable length.
    """

    # Defaults for instance variables
    type = None     # Node type (token if < 256, symbol if >= 256)
    content = None  # Optional content matching pattern
    name = None     # Optional name used to store match in results dict

    def __new__(cls, *args, **kwds):
        """Constructor that prevents BasePattern from being instantiated."""
        assert cls is not BasePattern, "Cannot instantiate BasePattern"
        return object.__new__(cls)

    def __repr__(self):
        args = [type_repr(self.type), self.content, self.name]
        while args and args[-1] is None:
            del args[-1]
        return "%s(%s)" % (self.__class__.__name__, ", ".join(map(repr, args)))

    def optimize(self):
        """
        A subclass can define this as a hook for optimizations.

        Returns either self or another node with the same effect.
        """
        return self

    def match(self, node, results=None):
        """
        Does this pattern exactly match a node?

        Returns True if it matches, False if not.

        If results is not None, it must be a dict which will be
        updated with the nodes matching named subpatterns.

        Default implementation for non-wildcard patterns.
        """
        if self.type is not None and node.type != self.type:
            return False
        if self.content is not None:
            r = None
            if results is not None:
                r = {}
            if not self._submatch(node, r):
                return False
            if r:
                results.update(r)
        if results is not None and self.name:
            results[self.name] = node
        return True

    def match_seq(self, nodes, results=None):
        """
        Does this pattern exactly match a sequence of nodes?

        Default implementation for non-wildcard patterns.
        """
        if len(nodes) != 1:
            return False
        return self.match(nodes[0], results)

    def generate_matches(self, nodes):
        """
        Generator yielding all matches for this pattern.

        Default implementation for non-wildcard patterns.
        """
        r = {}
        if nodes and self.match(nodes[0], r):
            yield 1, r


class LeafPattern(BasePattern):

    def __init__(self, type=None, content=None, name=None):
        """
        Initializer.  Takes optional type, content, and name.

        The type, if given must be a token type (< 256).  If not given,
        this matches any *leaf* node; the content may still be required.

        The content, if given, must be a string.

        If a name is given, the matching node is stored in the results
        dict under that key.
        """
        if type is not None:
            assert 0 <= type < 256, type
        if content is not None:
            assert isinstance(content, str), repr(content)
        self.type = type
        self.content = content
        self.name = name

    def match(self, node, results=None):
        """Override match() to insist on a leaf node."""
        if not isinstance(node, Leaf):
            return False
        return BasePattern.match(self, node, results)

    def _submatch(self, node, results=None):
        """
        Match the pattern's content to the node's children.

        This assumes the node type matches and self.content is not None.

        Returns True if it matches, False if not.

        If results is not None, it must be a dict which will be
        updated with the nodes matching named subpatterns.

        When returning False, the results dict may still be updated.
        """
        return self.content == node.value


class NodePattern(BasePattern):

    wildcards = False

    def __init__(self, type=None, content=None, name=None):
        """
        Initializer.  Takes optional type, content, and name.

        The type, if given, must be a symbol type (>= 256).  If the
        type is None this matches *any* single node (leaf or not),
        except if content is not None, in which it only matches
        non-leaf nodes that also match the content pattern.

        The content, if not None, must be a sequence of Patterns that
        must match the node's children exactly.  If the content is
        given, the type must not be None.

        If a name is given, the matching node is stored in the results
        dict under that key.
        """
        if type is not None:
            assert type >= 256, type
        if content is not None:
            assert not isinstance(content, str), repr(content)
            content = list(content)
            for i, item in enumerate(content):
                assert isinstance(item, BasePattern), (i, item)
                if isinstance(item, WildcardPattern):
                    self.wildcards = True
        self.type = type
        self.content = content
        self.name = name

    def _submatch(self, node, results=None):
        """
        Match the pattern's content to the node's children.

        This assumes the node type matches and self.content is not None.

        Returns True if it matches, False if not.

        If results is not None, it must be a dict which will be
        updated with the nodes matching named subpatterns.

        When returning False, the results dict may still be updated.
        """
        if self.wildcards:
            for c, r in generate_matches(self.content, node.children):
                if c == len(node.children):
                    if results is not None:
                        results.update(r)
                    return True
            return False
        if len(self.content) != len(node.children):
            return False
        for subpattern, child in zip(self.content, node.children):
            if not subpattern.match(child, results):
                return False
        return True


class WildcardPattern(BasePattern):

    """
    A wildcard pattern can match zero or more nodes.

    This has all the flexibility needed to implement patterns like:

    .*      .+      .?      .{m,n}
    (a b c | d e | f)
    (...)*  (...)+  (...)?  (...){m,n}

    except it always uses non-greedy matching.
    """

    def __init__(self, content=None, min=0, max=HUGE, name=None):
        """
        Initializer.

        Args:
            content: optional sequence of subsequences of patterns;
                     if absent, matches one node;
                     if present, each subsequence is an alternative [*]
            min: optional minimum number of times to match, default 0
            max: optional maximum number of times to match, default HUGE
            name: optional name assigned to this match

        [*] Thus, if content is [[a, b, c], [d, e], [f, g, h]] this is
            equivalent to (a b c | d e | f g h); if content is None,
            this is equivalent to '.' in regular expression terms.
            The min and max parameters work as follows:
                min=0, max=maxint: .*
                min=1, max=maxint: .+
                min=0, max=1: .?
                min=1, max=1: .
            If content is not None, replace the dot with the parenthesized
            list of alternatives, e.g. (a b c | d e | f g h)*
        """
        assert 0 <= min <= max <= HUGE, (min, max)
        if content is not None:
            content = tuple(map(tuple, content))  # Protect against alterations
            # Check sanity of alternatives
            assert len(content), repr(content)  # Can't have zero alternatives
            for alt in content:
                assert len(alt), repr(alt) # Can have empty alternatives
        self.content = content
        self.min = min
        self.max = max
        self.name = name

    def optimize(self):
        """Optimize certain stacked wildcard patterns."""
        subpattern = None
        if (self.content is not None and
            len(self.content) == 1 and len(self.content[0]) == 1):
            subpattern = self.content[0][0]
        if self.min == 1 and self.max == 1:
            if self.content is None:
                return NodePattern(name=self.name)
            if subpattern is not None and  self.name == subpattern.name:
                return subpattern.optimize()
        if (self.min <= 1 and isinstance(subpattern, WildcardPattern) and
            subpattern.min <= 1 and self.name == subpattern.name):
            return WildcardPattern(subpattern.content,
                                   self.min*subpattern.min,
                                   self.max*subpattern.max,
                                   subpattern.name)
        return self

    def match(self, node, results=None):
        """Does this pattern exactly match a node?"""
        return self.match_seq([node], results)

    def match_seq(self, nodes, results=None):
        """Does this pattern exactly match a sequence of nodes?"""
        for c, r in self.generate_matches(nodes):
            if c == len(nodes):
                if results is not None:
                    results.update(r)
                    if self.name:
                        results[self.name] = list(nodes)
                return True
        return False

    def generate_matches(self, nodes):
        """
        Generator yielding matches for a sequence of nodes.

        Args:
            nodes: sequence of nodes

        Yields:
            (count, results) tuples where:
            count: the match comprises nodes[:count];
            results: dict containing named submatches.
        """
        if self.content is None:
            # Shortcut for special case (see __init__.__doc__)
            for count in range(self.min, 1 + min(len(nodes), self.max)):
                r = {}
                if self.name:
                    r[self.name] = nodes[:count]
                yield count, r
        elif self.name == "bare_name":
            yield self._bare_name_matches(nodes)
        else:
            # The reason for this is that hitting the recursion limit usually
            # results in some ugly messages about how RuntimeErrors are being
            # ignored. We only have to do this on CPython, though, because other
            # implementations don't have this nasty bug in the first place.
            if hasattr(sys, "getrefcount"):
                save_stderr = sys.stderr
                sys.stderr = StringIO()
            try:
                for count, r in self._recursive_matches(nodes, 0):
                    if self.name:
                        r[self.name] = nodes[:count]
                    yield count, r
            except RuntimeError:
                # Fall back to the iterative pattern matching scheme if the
                # recursive scheme hits the recursion limit (RecursionError).
                for count, r in self._iterative_matches(nodes):
                    if self.name:
                        r[self.name] = nodes[:count]
                    yield count, r
            finally:
                if hasattr(sys, "getrefcount"):
                    sys.stderr = save_stderr

    def _iterative_matches(self, nodes):
        """Helper to iteratively yield the matches."""
        nodelen = len(nodes)
        if 0 >= self.min:
            yield 0, {}

        results = []
        # generate matches that use just one alt from self.content
        for alt in self.content:
            for c, r in generate_matches(alt, nodes):
                yield c, r
                results.append((c, r))

        # for each match, iterate down the nodes
        while results:
            new_results = []
            for c0, r0 in results:
                # stop if the entire set of nodes has been matched
                if c0 < nodelen and c0 <= self.max:
                    for alt in self.content:
                        for c1, r1 in generate_matches(alt, nodes[c0:]):
                            if c1 > 0:
                                r = {}
                                r.update(r0)
                                r.update(r1)
                                yield c0 + c1, r
                                new_results.append((c0 + c1, r))
            results = new_results

    def _bare_name_matches(self, nodes):
        """Special optimized matcher for bare_name."""
        count = 0
        r = {}
        done = False
        max = len(nodes)
        while not done and count < max:
            done = True
            for leaf in self.content:
                if leaf[0].match(nodes[count], r):
                    count += 1
                    done = False
                    break
        r[self.name] = nodes[:count]
        return count, r

    def _recursive_matches(self, nodes, count):
        """Helper to recursively yield the matches."""
        assert self.content is not None
        if count >= self.min:
            yield 0, {}
        if count < self.max:
            for alt in self.content:
                for c0, r0 in generate_matches(alt, nodes):
                    for c1, r1 in self._recursive_matches(nodes[c0:], count+1):
                        r = {}
                        r.update(r0)
                        r.update(r1)
                        yield c0 + c1, r


class NegatedPattern(BasePattern):

    def __init__(self, content=None):
        """
        Initializer.

        The argument is either a pattern or None.  If it is None, this
        only matches an empty sequence (effectively '$' in regex
        lingo).  If it is not None, this matches whenever the argument
        pattern doesn't have any matches.
        """
        if content is not None:
            assert isinstance(content, BasePattern), repr(content)
        self.content = content

    def match(self, node):
        # We never match a node in its entirety
        return False

    def match_seq(self, nodes):
        # We only match an empty sequence of nodes in its entirety
        return len(nodes) == 0

    def generate_matches(self, nodes):
        if self.content is None:
            # Return a match if there is an empty sequence
            if len(nodes) == 0:
                yield 0, {}
        else:
            # Return a match if the argument pattern has no matches
            for c, r in self.content.generate_matches(nodes):
                return
            yield 0, {}


def generate_matches(patterns, nodes):
    """
    Generator yielding matches for a sequence of patterns and nodes.

    Args:
        patterns: a sequence of patterns
        nodes: a sequence of nodes

    Yields:
        (count, results) tuples where:
        count: the entire sequence of patterns matches nodes[:count];
        results: dict containing named submatches.
        """
    if not patterns:
        yield 0, {}
    else:
        p, rest = patterns[0], patterns[1:]
        for c0, r0 in p.generate_matches(nodes):
            if not rest:
                yield c0, r0
            else:
                for c1, r1 in generate_matches(rest, nodes[c0:]):
                    r = {}
                    r.update(r0)
                    r.update(r1)
                    yield c0 + c1, r
