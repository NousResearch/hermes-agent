/* Module definition and import interface */

#ifndef Py_IMPORT_H
#define Py_IMPORT_H
#ifdef __cplusplus
extern "C" {
#endif

PyAPI_FUNC(long) PyImport_GetMagicNumber(void);
PyAPI_FUNC(const char *) PyImport_GetMagicTag(void);
PyAPI_FUNC(PyObject *) PyImport_ExecCodeModule(
    const char *name,           /* UTF-8 encoded string */
    PyObject *co
    );
PyAPI_FUNC(PyObject *) PyImport_ExecCodeModuleEx(
    const char *name,           /* UTF-8 encoded string */
    PyObject *co,
    const char *pathname        /* decoded from the filesystem encoding */
    );
PyAPI_FUNC(PyObject *) PyImport_ExecCodeModuleWithPathnames(
    const char *name,           /* UTF-8 encoded string */
    PyObject *co,
    const char *pathname,       /* decoded from the filesystem encoding */
    const char *cpathname       /* decoded from the filesystem encoding */
    );
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03030000
PyAPI_FUNC(PyObject *) PyImport_ExecCodeModuleObject(
    PyObject *name,
    PyObject *co,
    PyObject *pathname,
    PyObject *cpathname
    );
#endif
PyAPI_FUNC(PyObject *) PyImport_GetModuleDict(void);
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03070000
PyAPI_FUNC(PyObject *) PyImport_GetModule(PyObject *name);
#endif
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03030000
PyAPI_FUNC(PyObject *) PyImport_AddModuleObject(
    PyObject *name
    );
#endif
PyAPI_FUNC(PyObject *) PyImport_AddModule(
    const char *name            /* UTF-8 encoded string */
    );
PyAPI_FUNC(PyObject *) PyImport_ImportModule(
    const char *name            /* UTF-8 encoded string */
    );
PyAPI_FUNC(PyObject *) PyImport_ImportModuleNoBlock(
    const char *name            /* UTF-8 encoded string */
    );
PyAPI_FUNC(PyObject *) PyImport_ImportModuleLevel(
    const char *name,           /* UTF-8 encoded string */
    PyObject *globals,
    PyObject *locals,
    PyObject *fromlist,
    int level
    );
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03050000
PyAPI_FUNC(PyObject *) PyImport_ImportModuleLevelObject(
    PyObject *name,
    PyObject *globals,
    PyObject *locals,
    PyObject *fromlist,
    int level
    );
#endif

#define PyImport_ImportModuleEx(n, g, l, f) \
    PyImport_ImportModuleLevel(n, g, l, f, 0)

PyAPI_FUNC(PyObject *) PyImport_GetImporter(PyObject *path);
PyAPI_FUNC(PyObject *) PyImport_Import(PyObject *name);
PyAPI_FUNC(PyObject *) PyImport_ReloadModule(PyObject *m);
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03030000
PyAPI_FUNC(int) PyImport_ImportFrozenModuleObject(
    PyObject *name
    );
#endif
PyAPI_FUNC(int) PyImport_ImportFrozenModule(
    const char *name            /* UTF-8 encoded string */
    );

PyAPI_FUNC(int) PyImport_AppendInittab(
    const char *name,           /* ASCII encoded string */
    PyObject* (*initfunc)(void)
    );

#ifndef Py_LIMITED_API
#  define Py_CPYTHON_IMPORT_H
#  include "cpython/import.h"
#  undef Py_CPYTHON_IMPORT_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_IMPORT_H */
