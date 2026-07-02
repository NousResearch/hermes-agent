
/* Interfaces to configure, query, create & destroy the Python runtime */

#ifndef Py_PYLIFECYCLE_H
#define Py_PYLIFECYCLE_H
#ifdef __cplusplus
extern "C" {
#endif


/* Initialization and finalization */
PyAPI_FUNC(void) Py_Initialize(void);
PyAPI_FUNC(void) Py_InitializeEx(int);
PyAPI_FUNC(void) Py_Finalize(void);
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03060000
PyAPI_FUNC(int) Py_FinalizeEx(void);
#endif
PyAPI_FUNC(int) Py_IsInitialized(void);

/* Subinterpreter support */
PyAPI_FUNC(PyThreadState *) Py_NewInterpreter(void);
PyAPI_FUNC(void) Py_EndInterpreter(PyThreadState *);


/* Py_PyAtExit is for the atexit module, Py_AtExit is for low-level
 * exit functions.
 */
PyAPI_FUNC(int) Py_AtExit(void (*func)(void));

PyAPI_FUNC(void) _Py_NO_RETURN Py_Exit(int);

/* Bootstrap __main__ (defined in Modules/main.c) */
PyAPI_FUNC(int) Py_Main(int argc, wchar_t **argv);
PyAPI_FUNC(int) Py_BytesMain(int argc, char **argv);

/* In pathconfig.c */
Py_DEPRECATED(3.11) PyAPI_FUNC(void) Py_SetProgramName(const wchar_t *);
PyAPI_FUNC(wchar_t *) Py_GetProgramName(void);

Py_DEPRECATED(3.11) PyAPI_FUNC(void) Py_SetPythonHome(const wchar_t *);
PyAPI_FUNC(wchar_t *) Py_GetPythonHome(void);

PyAPI_FUNC(wchar_t *) Py_GetProgramFullPath(void);

PyAPI_FUNC(wchar_t *) Py_GetPrefix(void);
PyAPI_FUNC(wchar_t *) Py_GetExecPrefix(void);
PyAPI_FUNC(wchar_t *) Py_GetPath(void);
Py_DEPRECATED(3.11) PyAPI_FUNC(void) Py_SetPath(const wchar_t *);
#ifdef MS_WINDOWS
int _Py_CheckPython3(void);
#endif

/* In their own files */
PyAPI_FUNC(const char *) Py_GetVersion(void);
PyAPI_FUNC(const char *) Py_GetPlatform(void);
PyAPI_FUNC(const char *) Py_GetCopyright(void);
PyAPI_FUNC(const char *) Py_GetCompiler(void);
PyAPI_FUNC(const char *) Py_GetBuildInfo(void);

/* Signals */
typedef void (*PyOS_sighandler_t)(int);
PyAPI_FUNC(PyOS_sighandler_t) PyOS_getsig(int);
PyAPI_FUNC(PyOS_sighandler_t) PyOS_setsig(int, PyOS_sighandler_t);

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x030B0000
PyAPI_DATA(const unsigned long) Py_Version;
#endif

#ifndef Py_LIMITED_API
#  define Py_CPYTHON_PYLIFECYCLE_H
#  include "cpython/pylifecycle.h"
#  undef Py_CPYTHON_PYLIFECYCLE_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_PYLIFECYCLE_H */
