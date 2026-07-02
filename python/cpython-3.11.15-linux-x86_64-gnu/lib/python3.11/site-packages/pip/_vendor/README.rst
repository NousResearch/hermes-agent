================
Vendoring Policy
================

* Vendored libraries **MUST** not be modified except as required to
  successfully vendor them.
* Vendored libraries **MUST** be released copies of libraries available on
  PyPI.
* Vendored libraries **MUST** be available under a license that allows
  them to be integrated into ``pip``, which is released under the MIT license.
* Vendored libraries **MUST** be accompanied with LICENSE files.
* The versions of libraries vendored in pip **MUST** be reflected in
  ``pip/_vendor/vendor.txt``.
* Vendored libraries **MUST** function without any build steps such as ``2to3``
  or compilation of C code, practically this limits to single source 2.x/3.x and
  pure Python.
* Any modifications made to libraries **MUST** be noted in
  ``pip/_vendor/README.rst`` and their corresponding patches **MUST** be
  included ``tools/vendoring/patches``.
* Vendored libraries should have corresponding ``vendored()`` entries in
  ``pip/_vendor/__init__.py``.

Rationale
=========

Historically pip has not had any dependencies except for ``setuptools`` itself,
choosing instead to implement any functionality it needed to prevent needing
a dependency. However, starting with pip 1.5, we began to replace code that was
implemented inside of pip with reusable libraries from PyPI. This brought the
typical benefits of reusing libraries instead of reinventing the wheel like
higher quality and more battle tested code, centralization of bug fixes
(particularly security sensitive ones), and better/more features for less work.

However, there are several issues with having dependencies in the traditional
way (via ``install_requires``) for pip. These issues are:

**Fragility**
   When pip depends on another library to function then if for whatever reason
   that library either isn't installed or an incompatible version is installed
   then pip ceases to function. This is of course true for all Python
   applications, however for every application *except* for pip the way you fix
   it is by re-running pip. Obviously, when pip can't run, you can't use pip to
   fix pip, so you're left having to manually resolve dependencies and
   installing them by hand.

**Making other libraries uninstallable**
   One of pip's current dependencies is the ``requests`` library, for which pip
   requires a fairly recent version to run.  If pip depended on ``requests`` in
   the traditional manner, then we'd either have to maintain compatibility with
   every ``requests`` version that has ever existed (and ever will), OR allow
   pip to render certain versions of ``requests`` uninstallable. (The second
   issue, although technically true for any Python application, is magnified by
   pip's ubiquity; pip is installed by default in Python, in ``pyvenv``, and in
   ``virtualenv``.)

**Security**
   This might seem puzzling at first glance, since vendoring has a tendency to
   complicate updating dependencies for security updates, and that holds true
   for pip. However, given the *other* reasons for avoiding dependencies, the
   alternative is for pip to reinvent the wheel itself.  This is what pip did
   historically. It forced pip to re-implement its own HTTPS verification
   routines as a workaround for the Python standard library's lack of SSL
   validation, which resulted in similar bugs in the validation routine in
   ``requests`` and ``urllib3``, except that they had to be discovered and
   fixed independently. Even though we're vendoring, reusing libraries keeps
   pip more secure by relying on the great work of our dependencies, *and*
   allowing for faster, easier security fixes by simply pulling in newer
   versions of dependencies.

**Bootstrapping**
   Currently most popular methods of installing pip rely on pip's
   self-contained nature to install pip itself. These tools work by bundling a
   copy of pip, adding it to ``sys.path``, and then executing that copy of pip.
   This is done instead of implementing a "mini installer" (to reduce
   duplication); pip already knows how to install a Python package, and is far
   more battle-tested than any "mini installer" could ever possibly be.

Many downstream redistributors have policies against this kind of bundling, and
instead opt to patch the software they distribute to debundle it and make it
rely on the global versions of the software that they already have packaged
(which may have its own patches applied to it). We (the pip team) would prefer
it if pip was *not* debundled in this manner due to the above reasons and
instead we would prefer it if pip would be left intact as it is now.

In the longer term, if someone has a *portable* solution to the above problems,
other than the bundling method we currently use, that doesn't add additional
problems that are unreasonable then we would be happy to consider, and possibly
switch to said method. This solution must function correctly across all of the
situation that we expect pip to be used and not mandate some external mechanism
such as OS packages.


Modifications
=============

* ``setuptools`` is completely stripped to only keep ``pkg_resources``.
* ``pkg_resources`` has been modified to import its dependencies from
  ``pip._vendor``, and to use the vendored copy of ``platformdirs``
  rather than ``appdirs``.
* ``packaging`` has been modified to import its dependencies from
  ``pip._vendor``.
* ``CacheControl`` has been modified to import its dependencies from
  ``pip._vendor``.
* ``requests`` has been modified to import its other dependencies from
  ``pip._vendor`` and to *not* load ``simplejson`` (all platforms) and
  ``pyopenssl`` (Windows).
* ``platformdirs`` has been modified to import its submodules from ``pip._vendor.platformdirs``.

Automatic Vendoring
===================

Vendoring is automated via the `vendoring <https://pypi.org/project/vendoring/>`_ tool from the content of
``pip/_vendor/vendor.txt`` and the different patches in
``tools/vendoring/patches``.
Launch it via ``vendoring sync . -v`` (requires ``vendoring>=0.2.2``).
Tool configuration is done via ``pyproject.toml``.

To update the vendored library versions, we have a session defined in ``nox``.
The command to upgrade everything is::

    nox -s vendoring -- --upgrade-all --skip setuptools

We do not upgrade ``setuptools``, because we only rely on ``pkg_resources``,
and tracking every ``setuptools`` change is unnecessary for our needs.


Managing Local Patches
======================

The ``vendoring`` tool automatically applies our local patches, but updating,
the patches sometimes no longer apply cleanly. In that case, the update will
fail. To resolve this, take the following steps:

1. Revert any incomplete changes in the revendoring branch, to ensure you have
   a clean starting point.
2. Run the revendoring of the library with a problem again: ``nox -s vendoring
   -- --upgrade <library_name>``.
3. This will fail again, but you will have the original source in your working
   directory. Review the existing patch against the source, and modify the patch
   to reflect the new version of the source. If you ``git add`` the changes the
   vendoring made, you can modify the source to reflect the patch file and then
   generate a new patch with ``git diff``.
4. Now, revert everything *except* the patch file changes. Leave the modified
   patch file unstaged but saved in the working tree.
5. Re-run the vendoring. This time, it should pick up the changed patch file
   and apply it cleanly. The patch file changes will be committed along with the
   revendoring, so the new commit should be ready to test and publish as a PR.


Debundling
==========

As mentioned in the rationale, we, the pip team, would prefer it if pip was not
debundled (other than optionally ``pip/_vendor/requests/cacert.pem``) and that
pip was left intact. However, if you insist on doing so, we have a
semi-supported method (that we don't test in our CI) and requires a bit of
extra work on your end in order to solve the problems described above.

1. Delete everything in ``pip/_vendor/`` **except** for
   ``pip/_vendor/__init__.py`` and ``pip/_vendor/vendor.txt``.
2. Generate wheels for each of pip's dependencies (and any of their
   dependencies) using your patched copies of these libraries. These must be
   placed somewhere on the filesystem that pip can access (``pip/_vendor`` is
   the default assumption).
3. Modify ``pip/_vendor/__init__.py`` so that the ``DEBUNDLED`` variable is
   ``True``.
4. Upon installation, the ``INSTALLER`` file in pip's own ``dist-info``
   directory should be set to something other than ``pip``, so that pip
   can detect that it wasn't installed using itself.
5. *(optional)* If you've placed the wheels in a location other than
   ``pip/_vendor/``, then modify ``pip/_vendor/__init__.py`` so that the
   ``WHEEL_DIR`` variable points to the location you've placed them.
6. *(optional)* Update the ``pip_self_version_check`` logic to use the
   appropriate logic for determining the latest available version of pip and
   prompt the user with the correct upgrade message.

Note that partial debundling is **NOT** supported. You need to prepare wheels
for all dependencies for successful debundling.
