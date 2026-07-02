"""Fix incompatible imports and module references."""
# Authors: Collin Winter, Nick Edds

# Local imports
from .. import fixer_base
from ..fixer_util import Name, attr_chain

MAPPING = {'StringIO':  'io',
           'cStringIO': 'io',
           'cPickle': 'pickle',
           '__builtin__' : 'builtins',
           'copy_reg': 'copyreg',
           'Queue': 'queue',
           'SocketServer': 'socketserver',
           'ConfigParser': 'configparser',
           'repr': 'reprlib',
           'FileDialog': 'tkinter.filedialog',
           'tkFileDialog': 'tkinter.filedialog',
           'SimpleDialog': 'tkinter.simpledialog',
           'tkSimpleDialog': 'tkinter.simpledialog',
           'tkColorChooser': 'tkinter.colorchooser',
           'tkCommonDialog': 'tkinter.commondialog',
           'Dialog': 'tkinter.dialog',
           'Tkdnd': 'tkinter.dnd',
           'tkFont': 'tkinter.font',
           'tkMessageBox': 'tkinter.messagebox',
           'ScrolledText': 'tkinter.scrolledtext',
           'Tkconstants': 'tkinter.constants',
           'Tix': 'tkinter.tix',
           'ttk': 'tkinter.ttk',
           'Tkinter': 'tkinter',
           'markupbase': '_markupbase',
           '_winreg': 'winreg',
           'thread': '_thread',
           'dummy_thread': '_dummy_thread',
           # anydbm and whichdb are handled by fix_imports2
           'dbhash': 'dbm.bsd',
           'dumbdbm': 'dbm.dumb',
           'dbm': 'dbm.ndbm',
           'gdbm': 'dbm.gnu',
           'xmlrpclib': 'xmlrpc.client',
           'DocXMLRPCServer': 'xmlrpc.server',
           'SimpleXMLRPCServer': 'xmlrpc.server',
           'httplib': 'http.client',
           'htmlentitydefs' : 'html.entities',
           'HTMLParser' : 'html.parser',
           'Cookie': 'http.cookies',
           'cookielib': 'http.cookiejar',
           'BaseHTTPServer': 'http.server',
           'SimpleHTTPServer': 'http.server',
           'CGIHTTPServer': 'http.server',
           #'test.test_support': 'test.support',
           'commands': 'subprocess',
           'UserString' : 'collections',
           'UserList' : 'collections',
           'urlparse' : 'urllib.parse',
           'robotparser' : 'urllib.robotparser',
}


def alternates(members):
    return "(" + "|".join(map(repr, members)) + ")"


def build_pattern(mapping=MAPPING):
    mod_list = ' | '.join(["module_name='%s'" % key for key in mapping])
    bare_names = alternates(mapping.keys())

    yield """name_import=import_name< 'import' ((%s) |
               multiple_imports=dotted_as_names< any* (%s) any* >) >
          """ % (mod_list, mod_list)
    yield """import_from< 'from' (%s) 'import' ['(']
              ( any | import_as_name< any 'as' any > |
                import_as_names< any* >)  [')'] >
          """ % mod_list
    yield """import_name< 'import' (dotted_as_name< (%s) 'as' any > |
               multiple_imports=dotted_as_names<
                 any* dotted_as_name< (%s) 'as' any > any* >) >
          """ % (mod_list, mod_list)

    # Find usages of module members in code e.g. thread.foo(bar)
    yield "power< bare_with_attr=(%s) trailer<'.' any > any* >" % bare_names


class FixImports(fixer_base.BaseFix):

    BM_compatible = True
    keep_line_order = True
    # This is overridden in fix_imports2.
    mapping = MAPPING

    # We want to run this fixer late, so fix_import doesn't try to make stdlib
    # renames into relative imports.
    run_order = 6

    def build_pattern(self):
        return "|".join(build_pattern(self.mapping))

    def compile_pattern(self):
        # We override this, so MAPPING can be pragmatically altered and the
        # changes will be reflected in PATTERN.
        self.PATTERN = self.build_pattern()
        super(FixImports, self).compile_pattern()

    # Don't match the node if it's within another match.
    def match(self, node):
        match = super(FixImports, self).match
        results = match(node)
        if results:
            # Module usage could be in the trailer of an attribute lookup, so we
            # might have nested matches when "bare_with_attr" is present.
            if "bare_with_attr" not in results and \
                    any(match(obj) for obj in attr_chain(node, "parent")):
                return False
            return results
        return False

    def start_tree(self, tree, filename):
        super(FixImports, self).start_tree(tree, filename)
        self.replace = {}

    def transform(self, node, results):
        import_mod = results.get("module_name")
        if import_mod:
            mod_name = import_mod.value
            new_name = self.mapping[mod_name]
            import_mod.replace(Name(new_name, prefix=import_mod.prefix))
            if "name_import" in results:
                # If it's not a "from x import x, y" or "import x as y" import,
                # marked its usage to be replaced.
                self.replace[mod_name] = new_name
            if "multiple_imports" in results:
                # This is a nasty hack to fix multiple imports on a line (e.g.,
                # "import StringIO, urlparse"). The problem is that I can't
                # figure out an easy way to make a pattern recognize the keys of
                # MAPPING randomly sprinkled in an import statement.
                results = self.match(node)
                if results:
                    self.transform(node, results)
        else:
            # Replace usage of the module.
            bare_name = results["bare_with_attr"][0]
            new_name = self.replace.get(bare_name.value)
            if new_name:
                bare_name.replace(Name(new_name, prefix=bare_name.prefix))
