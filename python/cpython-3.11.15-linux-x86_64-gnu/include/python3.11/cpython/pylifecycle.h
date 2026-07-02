#ifndef Py_CPYTHON_PYLIFECYCLE_H
#  error "this header file must not be included directly"
#endif

/* Py_FrozenMain is kept out of the Limited API until documented and present
   in all builds of Python */
PyAPI_FUNC(int) Py_FrozenMain(int argc, char **argv);

/* Only used by applications that embed the interpreter and need to
 * override the standard encoding determination mechanism
 */
Py_DEPRECATED(3.11) PyAPI_FUNC(int) Py_SetStandardStreamEncoding(
    const char *encoding,
    const char *errors);

/* PEP 432 Multi-phase initialization API (Private while provisional!) */

PyAPI_FUNC(PyStatus) Py_PreInitialize(
    const PyPreConfig *src_config);
PyAPI_FUNC(PyStatus) Py_PreInitializeFromBytesArgs(
    const PyPreConfig *src_config,
    Py_ssize_t argc,
    char **argv);
PyAPI_FUNC(PyStatus) Py_PreInitializeFromArgs(
    const PyPreConfig *src_config,
    Py_ssize_t argc,
    wchar_t **argv);

PyAPI_FUNC(int) _Py_IsCoreInitialized(void);


/* Initialization and finalization */

PyAPI_FUNC(PyStatus) Py_InitializeFromConfig(
    const PyConfig *config);
PyAPI_FUNC(PyStatus) _Py_InitializeMain(void);

PyAPI_FUNC(int) Py_RunMain(void);


PyAPI_FUNC(void) _Py_NO_RETURN Py_ExitStatusException(PyStatus err);

/* Restore signals that the interpreter has called SIG_IGN on to SIG_DFL. */
PyAPI_FUNC(void) _Py_RestoreSignals(void);

PyAPI_FUNC(int) Py_FdIsInteractive(FILE *, const char *);
PyAPI_FUNC(int) _Py_FdIsInteractive(FILE *fp, PyObject *filename);

Py_DEPRECATED(3.11) PyAPI_FUNC(void) _Py_SetProgramFullPath(const wchar_t *);

PyAPI_FUNC(const char *) _Py_gitidentifier(void);
PyAPI_FUNC(const char *) _Py_gitversion(void);

PyAPI_FUNC(int) _Py_IsFinalizing(void);

/* Random */
PyAPI_FUNC(int) _PyOS_URandom(void *buffer, Py_ssize_t size);
PyAPI_FUNC(int) _PyOS_URandomNonblock(void *buffer, Py_ssize_t size);

/* Legacy locale support */
PyAPI_FUNC(int) _Py_CoerceLegacyLocale(int warn);
PyAPI_FUNC(int) _Py_LegacyLocaleDetected(int warn);
PyAPI_FUNC(char *) _Py_SetLocaleFromEnv(int category);

PyAPI_FUNC(PyThreadState *) _Py_NewInterpreter(int isolated_subinterpreter);
