"""Fix incompatible renames

Fixes:
  * sys.maxint -> sys.maxsize
"""
# Author: Christian Heimes
# based on Collin Winter's fix_import

# Local imports
from .. import fixer_base
from ..fixer_util import Name, attr_chain

MAPPING = {"sys":  {"maxint" : "maxsize"},
          }
LOOKUP = {}

def alternates(members):
    return "(" + "|".join(map(repr, members)) + ")"


def build_pattern():
    #bare = set()
    for module, replace in list(MAPPING.items()):
        for old_attr, new_attr in list(replace.items()):
            LOOKUP[(module, old_attr)] = new_attr
            #bare.add(module)
            #bare.add(old_attr)
            #yield """
            #      import_name< 'import' (module=%r
            #          | dotted_as_names< any* module=%r any* >) >
            #      """ % (module, module)
            yield """
                  import_from< 'from' module_name=%r 'import'
                      ( attr_name=%r | import_as_name< attr_name=%r 'as' any >) >
                  """ % (module, old_attr, old_attr)
            yield """
                  power< module_name=%r trailer< '.' attr_name=%r > any* >
                  """ % (module, old_attr)
    #yield """bare_name=%s""" % alternates(bare)


class FixRenames(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "|".join(build_pattern())

    order = "pre" # Pre-order tree traversal

    # Don't match the node if it's within another match
    def match(self, node):
        match = super(FixRenames, self).match
        results = match(node)
        if results:
            if any(match(obj) for obj in attr_chain(node, "parent")):
                return False
            return results
        return False

    #def start_tree(self, tree, filename):
    #    super(FixRenames, self).start_tree(tree, filename)
    #    self.replace = {}

    def transform(self, node, results):
        mod_name = results.get("module_name")
        attr_name = results.get("attr_name")
        #bare_name = results.get("bare_name")
        #import_mod = results.get("module")

        if mod_name and attr_name:
            new_attr = LOOKUP[(mod_name.value, attr_name.value)]
            attr_name.replace(Name(new_attr, prefix=attr_name.prefix))
