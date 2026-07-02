# Copyright 2006 Google, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Fixer for execfile.

This converts usages of the execfile function into calls to the built-in
exec() function.
"""

from .. import fixer_base
from ..fixer_util import (Comma, Name, Call, LParen, RParen, Dot, Node,
                          ArgList, String, syms)


class FixExecfile(fixer_base.BaseFix):
    BM_compatible = True

    PATTERN = """
    power< 'execfile' trailer< '(' arglist< filename=any [',' globals=any [',' locals=any ] ] > ')' > >
    |
    power< 'execfile' trailer< '(' filename=any ')' > >
    """

    def transform(self, node, results):
        assert results
        filename = results["filename"]
        globals = results.get("globals")
        locals = results.get("locals")

        # Copy over the prefix from the right parentheses end of the execfile
        # call.
        execfile_paren = node.children[-1].children[-1].clone()
        # Construct open().read().
        open_args = ArgList([filename.clone(), Comma(), String('"rb"', ' ')],
                            rparen=execfile_paren)
        open_call = Node(syms.power, [Name("open"), open_args])
        read = [Node(syms.trailer, [Dot(), Name('read')]),
                Node(syms.trailer, [LParen(), RParen()])]
        open_expr = [open_call] + read
        # Wrap the open call in a compile call. This is so the filename will be
        # preserved in the execed code.
        filename_arg = filename.clone()
        filename_arg.prefix = " "
        exec_str = String("'exec'", " ")
        compile_args = open_expr + [Comma(), filename_arg, Comma(), exec_str]
        compile_call = Call(Name("compile"), compile_args, "")
        # Finally, replace the execfile call with an exec call.
        args = [compile_call]
        if globals is not None:
            args.extend([Comma(), globals.clone()])
        if locals is not None:
            args.extend([Comma(), locals.clone()])
        return Call(Name("exec"), args, prefix=node.prefix)
