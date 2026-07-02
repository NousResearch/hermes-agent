"""Fixer for __metaclass__ = X -> (metaclass=X) methods.

   The various forms of classef (inherits nothing, inherits once, inherits
   many) don't parse the same in the CST so we look at ALL classes for
   a __metaclass__ and if we find one normalize the inherits to all be
   an arglist.

   For one-liner classes ('class X: pass') there is no indent/dedent so
   we normalize those into having a suite.

   Moving the __metaclass__ into the classdef can also cause the class
   body to be empty so there is some special casing for that as well.

   This fixer also tries very hard to keep original indenting and spacing
   in all those corner cases.

"""
# Author: Jack Diederich

# Local imports
from .. import fixer_base
from ..pygram import token
from ..fixer_util import syms, Node, Leaf


def has_metaclass(parent):
    """ we have to check the cls_node without changing it.
        There are two possibilities:
          1)  clsdef => suite => simple_stmt => expr_stmt => Leaf('__meta')
          2)  clsdef => simple_stmt => expr_stmt => Leaf('__meta')
    """
    for node in parent.children:
        if node.type == syms.suite:
            return has_metaclass(node)
        elif node.type == syms.simple_stmt and node.children:
            expr_node = node.children[0]
            if expr_node.type == syms.expr_stmt and expr_node.children:
                left_side = expr_node.children[0]
                if isinstance(left_side, Leaf) and \
                        left_side.value == '__metaclass__':
                    return True
    return False


def fixup_parse_tree(cls_node):
    """ one-line classes don't get a suite in the parse tree so we add
        one to normalize the tree
    """
    for node in cls_node.children:
        if node.type == syms.suite:
            # already in the preferred format, do nothing
            return

    # !%@#! one-liners have no suite node, we have to fake one up
    for i, node in enumerate(cls_node.children):
        if node.type == token.COLON:
            break
    else:
        raise ValueError("No class suite and no ':'!")

    # move everything into a suite node
    suite = Node(syms.suite, [])
    while cls_node.children[i+1:]:
        move_node = cls_node.children[i+1]
        suite.append_child(move_node.clone())
        move_node.remove()
    cls_node.append_child(suite)
    node = suite


def fixup_simple_stmt(parent, i, stmt_node):
    """ if there is a semi-colon all the parts count as part of the same
        simple_stmt.  We just want the __metaclass__ part so we move
        everything after the semi-colon into its own simple_stmt node
    """
    for semi_ind, node in enumerate(stmt_node.children):
        if node.type == token.SEMI: # *sigh*
            break
    else:
        return

    node.remove() # kill the semicolon
    new_expr = Node(syms.expr_stmt, [])
    new_stmt = Node(syms.simple_stmt, [new_expr])
    while stmt_node.children[semi_ind:]:
        move_node = stmt_node.children[semi_ind]
        new_expr.append_child(move_node.clone())
        move_node.remove()
    parent.insert_child(i, new_stmt)
    new_leaf1 = new_stmt.children[0].children[0]
    old_leaf1 = stmt_node.children[0].children[0]
    new_leaf1.prefix = old_leaf1.prefix


def remove_trailing_newline(node):
    if node.children and node.children[-1].type == token.NEWLINE:
        node.children[-1].remove()


def find_metas(cls_node):
    # find the suite node (Mmm, sweet nodes)
    for node in cls_node.children:
        if node.type == syms.suite:
            break
    else:
        raise ValueError("No class suite!")

    # look for simple_stmt[ expr_stmt[ Leaf('__metaclass__') ] ]
    for i, simple_node in list(enumerate(node.children)):
        if simple_node.type == syms.simple_stmt and simple_node.children:
            expr_node = simple_node.children[0]
            if expr_node.type == syms.expr_stmt and expr_node.children:
                # Check if the expr_node is a simple assignment.
                left_node = expr_node.children[0]
                if isinstance(left_node, Leaf) and \
                        left_node.value == '__metaclass__':
                    # We found an assignment to __metaclass__.
                    fixup_simple_stmt(node, i, simple_node)
                    remove_trailing_newline(simple_node)
                    yield (node, i, simple_node)


def fixup_indent(suite):
    """ If an INDENT is followed by a thing with a prefix then nuke the prefix
        Otherwise we get in trouble when removing __metaclass__ at suite start
    """
    kids = suite.children[::-1]
    # find the first indent
    while kids:
        node = kids.pop()
        if node.type == token.INDENT:
            break

    # find the first Leaf
    while kids:
        node = kids.pop()
        if isinstance(node, Leaf) and node.type != token.DEDENT:
            if node.prefix:
                node.prefix = ''
            return
        else:
            kids.extend(node.children[::-1])


class FixMetaclass(fixer_base.BaseFix):
    BM_compatible = True

    PATTERN = """
    classdef<any*>
    """

    def transform(self, node, results):
        if not has_metaclass(node):
            return

        fixup_parse_tree(node)

        # find metaclasses, keep the last one
        last_metaclass = None
        for suite, i, stmt in find_metas(node):
            last_metaclass = stmt
            stmt.remove()

        text_type = node.children[0].type # always Leaf(nnn, 'class')

        # figure out what kind of classdef we have
        if len(node.children) == 7:
            # Node(classdef, ['class', 'name', '(', arglist, ')', ':', suite])
            #                 0        1       2    3        4    5    6
            if node.children[3].type == syms.arglist:
                arglist = node.children[3]
            # Node(classdef, ['class', 'name', '(', 'Parent', ')', ':', suite])
            else:
                parent = node.children[3].clone()
                arglist = Node(syms.arglist, [parent])
                node.set_child(3, arglist)
        elif len(node.children) == 6:
            # Node(classdef, ['class', 'name', '(',  ')', ':', suite])
            #                 0        1       2     3    4    5
            arglist = Node(syms.arglist, [])
            node.insert_child(3, arglist)
        elif len(node.children) == 4:
            # Node(classdef, ['class', 'name', ':', suite])
            #                 0        1       2    3
            arglist = Node(syms.arglist, [])
            node.insert_child(2, Leaf(token.RPAR, ')'))
            node.insert_child(2, arglist)
            node.insert_child(2, Leaf(token.LPAR, '('))
        else:
            raise ValueError("Unexpected class definition")

        # now stick the metaclass in the arglist
        meta_txt = last_metaclass.children[0].children[0]
        meta_txt.value = 'metaclass'
        orig_meta_prefix = meta_txt.prefix

        if arglist.children:
            arglist.append_child(Leaf(token.COMMA, ','))
            meta_txt.prefix = ' '
        else:
            meta_txt.prefix = ''

        # compact the expression "metaclass = Meta" -> "metaclass=Meta"
        expr_stmt = last_metaclass.children[0]
        assert expr_stmt.type == syms.expr_stmt
        expr_stmt.children[1].prefix = ''
        expr_stmt.children[2].prefix = ''

        arglist.append_child(last_metaclass)

        fixup_indent(suite)

        # check for empty suite
        if not suite.children:
            # one-liner that was just __metaclass_
            suite.remove()
            pass_leaf = Leaf(text_type, 'pass')
            pass_leaf.prefix = orig_meta_prefix
            node.append_child(pass_leaf)
            node.append_child(Leaf(token.NEWLINE, '\n'))

        elif len(suite.children) > 1 and \
                 (suite.children[-2].type == token.INDENT and
                  suite.children[-1].type == token.DEDENT):
            # there was only one line in the class body and it was __metaclass__
            pass_leaf = Leaf(text_type, 'pass')
            suite.insert_child(-1, pass_leaf)
            suite.insert_child(-1, Leaf(token.NEWLINE, '\n'))
