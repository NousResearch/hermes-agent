"""Fixer for import statements.
If spam is being imported from the local directory, this import:
    from spam import eggs
Becomes:
    from .spam import eggs

And this import:
    import spam
Becomes:
    from . import spam
"""

# Local imports
from .. import fixer_base
from os.path import dirname, join, exists, sep
from ..fixer_util import FromImport, syms, token


def traverse_imports(names):
    """
    Walks over all the names imported in a dotted_as_names node.
    """
    pending = [names]
    while pending:
        node = pending.pop()
        if node.type == token.NAME:
            yield node.value
        elif node.type == syms.dotted_name:
            yield "".join([ch.value for ch in node.children])
        elif node.type == syms.dotted_as_name:
            pending.append(node.children[0])
        elif node.type == syms.dotted_as_names:
            pending.extend(node.children[::-2])
        else:
            raise AssertionError("unknown node type")


class FixImport(fixer_base.BaseFix):
    BM_compatible = True

    PATTERN = """
    import_from< 'from' imp=any 'import' ['('] any [')'] >
    |
    import_name< 'import' imp=any >
    """

    def start_tree(self, tree, name):
        super(FixImport, self).start_tree(tree, name)
        self.skip = "absolute_import" in tree.future_features

    def transform(self, node, results):
        if self.skip:
            return
        imp = results['imp']

        if node.type == syms.import_from:
            # Some imps are top-level (eg: 'import ham')
            # some are first level (eg: 'import ham.eggs')
            # some are third level (eg: 'import ham.eggs as spam')
            # Hence, the loop
            while not hasattr(imp, 'value'):
                imp = imp.children[0]
            if self.probably_a_local_import(imp.value):
                imp.value = "." + imp.value
                imp.changed()
        else:
            have_local = False
            have_absolute = False
            for mod_name in traverse_imports(imp):
                if self.probably_a_local_import(mod_name):
                    have_local = True
                else:
                    have_absolute = True
            if have_absolute:
                if have_local:
                    # We won't handle both sibling and absolute imports in the
                    # same statement at the moment.
                    self.warning(node, "absolute and local imports together")
                return

            new = FromImport(".", [imp])
            new.prefix = node.prefix
            return new

    def probably_a_local_import(self, imp_name):
        if imp_name.startswith("."):
            # Relative imports are certainly not local imports.
            return False
        imp_name = imp_name.split(".", 1)[0]
        base_path = dirname(self.filename)
        base_path = join(base_path, imp_name)
        # If there is no __init__.py next to the file its not in a package
        # so can't be a relative import.
        if not exists(join(dirname(base_path), "__init__.py")):
            return False
        for ext in [".py", sep, ".pyc", ".so", ".sl", ".pyd"]:
            if exists(base_path + ext):
                return True
        return False
