"""Fix changes imports of urllib which are now incompatible.
   This is rather similar to fix_imports, but because of the more
   complex nature of the fixing for urllib, it has its own fixer.
"""
# Author: Nick Edds

# Local imports
from lib2to3.fixes.fix_imports import alternates, FixImports
from lib2to3.fixer_util import (Name, Comma, FromImport, Newline,
                                find_indentation, Node, syms)

MAPPING = {"urllib":  [
                ("urllib.request",
                    ["URLopener", "FancyURLopener", "urlretrieve",
                     "_urlopener", "urlopen", "urlcleanup",
                     "pathname2url", "url2pathname", "getproxies"]),
                ("urllib.parse",
                    ["quote", "quote_plus", "unquote", "unquote_plus",
                     "urlencode", "splitattr", "splithost", "splitnport",
                     "splitpasswd", "splitport", "splitquery", "splittag",
                     "splittype", "splituser", "splitvalue", ]),
                ("urllib.error",
                    ["ContentTooShortError"])],
           "urllib2" : [
                ("urllib.request",
                    ["urlopen", "install_opener", "build_opener",
                     "Request", "OpenerDirector", "BaseHandler",
                     "HTTPDefaultErrorHandler", "HTTPRedirectHandler",
                     "HTTPCookieProcessor", "ProxyHandler",
                     "HTTPPasswordMgr",
                     "HTTPPasswordMgrWithDefaultRealm",
                     "AbstractBasicAuthHandler",
                     "HTTPBasicAuthHandler", "ProxyBasicAuthHandler",
                     "AbstractDigestAuthHandler",
                     "HTTPDigestAuthHandler", "ProxyDigestAuthHandler",
                     "HTTPHandler", "HTTPSHandler", "FileHandler",
                     "FTPHandler", "CacheFTPHandler",
                     "UnknownHandler"]),
                ("urllib.error",
                    ["URLError", "HTTPError"]),
           ]
}

# Duplicate the url parsing functions for urllib2.
MAPPING["urllib2"].append(MAPPING["urllib"][1])


def build_pattern():
    bare = set()
    for old_module, changes in MAPPING.items():
        for change in changes:
            new_module, members = change
            members = alternates(members)
            yield """import_name< 'import' (module=%r
                                  | dotted_as_names< any* module=%r any* >) >
                  """ % (old_module, old_module)
            yield """import_from< 'from' mod_member=%r 'import'
                       ( member=%s | import_as_name< member=%s 'as' any > |
                         import_as_names< members=any*  >) >
                  """ % (old_module, members, members)
            yield """import_from< 'from' module_star=%r 'import' star='*' >
                  """ % old_module
            yield """import_name< 'import'
                                  dotted_as_name< module_as=%r 'as' any > >
                  """ % old_module
            # bare_with_attr has a special significance for FixImports.match().
            yield """power< bare_with_attr=%r trailer< '.' member=%s > any* >
                  """ % (old_module, members)


class FixUrllib(FixImports):

    def build_pattern(self):
        return "|".join(build_pattern())

    def transform_import(self, node, results):
        """Transform for the basic import case. Replaces the old
           import name with a comma separated list of its
           replacements.
        """
        import_mod = results.get("module")
        pref = import_mod.prefix

        names = []

        # create a Node list of the replacement modules
        for name in MAPPING[import_mod.value][:-1]:
            names.extend([Name(name[0], prefix=pref), Comma()])
        names.append(Name(MAPPING[import_mod.value][-1][0], prefix=pref))
        import_mod.replace(names)

    def transform_member(self, node, results):
        """Transform for imports of specific module elements. Replaces
           the module to be imported from with the appropriate new
           module.
        """
        mod_member = results.get("mod_member")
        pref = mod_member.prefix
        member = results.get("member")

        # Simple case with only a single member being imported
        if member:
            # this may be a list of length one, or just a node
            if isinstance(member, list):
                member = member[0]
            new_name = None
            for change in MAPPING[mod_member.value]:
                if member.value in change[1]:
                    new_name = change[0]
                    break
            if new_name:
                mod_member.replace(Name(new_name, prefix=pref))
            else:
                self.cannot_convert(node, "This is an invalid module element")

        # Multiple members being imported
        else:
            # a dictionary for replacements, order matters
            modules = []
            mod_dict = {}
            members = results["members"]
            for member in members:
                # we only care about the actual members
                if member.type == syms.import_as_name:
                    as_name = member.children[2].value
                    member_name = member.children[0].value
                else:
                    member_name = member.value
                    as_name = None
                if member_name != ",":
                    for change in MAPPING[mod_member.value]:
                        if member_name in change[1]:
                            if change[0] not in mod_dict:
                                modules.append(change[0])
                            mod_dict.setdefault(change[0], []).append(member)

            new_nodes = []
            indentation = find_indentation(node)
            first = True
            def handle_name(name, prefix):
                if name.type == syms.import_as_name:
                    kids = [Name(name.children[0].value, prefix=prefix),
                            name.children[1].clone(),
                            name.children[2].clone()]
                    return [Node(syms.import_as_name, kids)]
                return [Name(name.value, prefix=prefix)]
            for module in modules:
                elts = mod_dict[module]
                names = []
                for elt in elts[:-1]:
                    names.extend(handle_name(elt, pref))
                    names.append(Comma())
                names.extend(handle_name(elts[-1], pref))
                new = FromImport(module, names)
                if not first or node.parent.prefix.endswith(indentation):
                    new.prefix = indentation
                new_nodes.append(new)
                first = False
            if new_nodes:
                nodes = []
                for new_node in new_nodes[:-1]:
                    nodes.extend([new_node, Newline()])
                nodes.append(new_nodes[-1])
                node.replace(nodes)
            else:
                self.cannot_convert(node, "All module elements are invalid")

    def transform_dot(self, node, results):
        """Transform for calls to module members in code."""
        module_dot = results.get("bare_with_attr")
        member = results.get("member")
        new_name = None
        if isinstance(member, list):
            member = member[0]
        for change in MAPPING[module_dot.value]:
            if member.value in change[1]:
                new_name = change[0]
                break
        if new_name:
            module_dot.replace(Name(new_name,
                                    prefix=module_dot.prefix))
        else:
            self.cannot_convert(node, "This is an invalid module element")

    def transform(self, node, results):
        if results.get("module"):
            self.transform_import(node, results)
        elif results.get("mod_member"):
            self.transform_member(node, results)
        elif results.get("bare_with_attr"):
            self.transform_dot(node, results)
        # Renaming and star imports are not supported for these modules.
        elif results.get("module_star"):
            self.cannot_convert(node, "Cannot handle star imports.")
        elif results.get("module_as"):
            self.cannot_convert(node, "This module is now multiple modules")
