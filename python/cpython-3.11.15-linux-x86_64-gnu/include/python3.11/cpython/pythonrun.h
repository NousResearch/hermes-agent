#ifndef Py_CPYTHON_PYTHONRUN_H
#  error "this header file must not be included directly"
#endif

PyAPI_FUNC(int) PyRun_SimpleStringFlags(const char *, PyCompilerFlags *);
PyAPI_FUNC(int) _PyRun_SimpleFileObject(
    FILE *fp,
    PyObject *filename,
    int closeit,
    PyCompilerFlags *flags);
PyAPI_FUNC(int) PyRun_AnyFileExFlags(
    FILE *fp,
    const char *filename,       /* decoded from the filesystem encoding */
    int closeit,
    PyCompilerFlags *flags);
PyAPI_FUNC(int) _PyRun_AnyFileObject(
    FILE *fp,
    PyObject *filename,
    int closeit,
    PyCompilerFlags *flags);
PyAPI_FUNC(int) PyRun_SimpleFileExFlags(
    FILE *fp,
    const char *filename,       /* decoded from the filesystem encoding */
    int closeit,
    PyCompilerFlags *flags);
PyAPI_FUNC(int) PyRun_InteractiveOneFlags(
    FILE *fp,
    const char *filename,       /* decoded from the filesystem encoding */
    PyCompilerFlags *flags);
PyAPI_FUNC(int) PyRun_InteractiveOneObject(
    FILE *fp,
    PyObject *filename,
    PyCompilerFlags *flags);
PyAPI_FUNC(int) PyRun_InteractiveLoopFlags(
    FILE *fp,
    const char *filename,       /* decoded from the filesystem encoding */
    PyCompilerFlags *flags);
PyAPI_FUNC(int) _PyRun_InteractiveLoopObject(
    FILE *fp,
    PyObject *filename,
    PyCompilerFlags *flags);


PyAPI_FUNC(PyObject *) PyRun_StringFlags(const char *, int, PyObject *,
                                         PyObject *, PyCompilerFlags *);

PyAPI_FUNC(PyObject *) PyRun_FileExFlags(
    FILE *fp,
    const char *filename,       /* decoded from the filesystem encoding */
    int start,
    PyObject *globals,
    PyObject *locals,
    int closeit,
    PyCompilerFlags *flags);


PyAPI_FUNC(PyObject *) Py_CompileStringExFlags(
    const char *str,
    const char *filename,       /* decoded from the filesystem encoding */
    int start,
    PyCompilerFlags *flags,
    int optimize);
PyAPI_FUNC(PyObject *) Py_CompileStringObject(
    const char *str,
    PyObject *filename, int start,
    PyCompilerFlags *flags,
    int optimize);

#define Py_CompileString(str, p, s) Py_CompileStringExFlags(str, p, s, NULL, -1)
#define Py_CompileStringFlags(str, p, s, f) Py_CompileStringExFlags(str, p, s, f, -1)


PyAPI_FUNC(const char *) _Py_SourceAsString(
    PyObject *cmd,
    const char *funcname,
    const char *what,
    PyCompilerFlags *cf,
    PyObject **cmd_copy);


/* A function flavor is also exported by libpython. It is required when
    libpython is accessed directly rather than using header files which defines
    macros below. On Windows, for example, PyAPI_FUNC() uses dllexport to
    export functions in pythonXX.dll. */
PyAPI_FUNC(PyObject *) PyRun_String(const char *str, int s, PyObject *g, PyObject *l);
PyAPI_FUNC(int) PyRun_AnyFile(FILE *fp, const char *name);
PyAPI_FUNC(int) PyRun_AnyFileEx(FILE *fp, const char *name, int closeit);
PyAPI_FUNC(int) PyRun_AnyFileFlags(FILE *, const char *, PyCompilerFlags *);
PyAPI_FUNC(int) PyRun_SimpleString(const char *s);
PyAPI_FUNC(int) PyRun_SimpleFile(FILE *f, const char *p);
PyAPI_FUNC(int) PyRun_SimpleFileEx(FILE *f, const char *p, int c);
PyAPI_FUNC(int) PyRun_InteractiveOne(FILE *f, const char *p);
PyAPI_FUNC(int) PyRun_InteractiveLoop(FILE *f, const char *p);
PyAPI_FUNC(PyObject *) PyRun_File(FILE *fp, const char *p, int s, PyObject *g, PyObject *l);
PyAPI_FUNC(PyObject *) PyRun_FileEx(FILE *fp, const char *p, int s, PyObject *g, PyObject *l, int c);
PyAPI_FUNC(PyObject *) PyRun_FileFlags(FILE *fp, const char *p, int s, PyObject *g, PyObject *l, PyCompilerFlags *flags);

/* Use macros for a bunch of old variants */
#define PyRun_String(str, s, g, l) PyRun_StringFlags(str, s, g, l, NULL)
#define PyRun_AnyFile(fp, name) PyRun_AnyFileExFlags(fp, name, 0, NULL)
#define PyRun_AnyFileEx(fp, name, closeit) \
    PyRun_AnyFileExFlags(fp, name, closeit, NULL)
#define PyRun_AnyFileFlags(fp, name, flags) \
    PyRun_AnyFileExFlags(fp, name, 0, flags)
#define PyRun_SimpleString(s) PyRun_SimpleStringFlags(s, NULL)
#define PyRun_SimpleFile(f, p) PyRun_SimpleFileExFlags(f, p, 0, NULL)
#define PyRun_SimpleFileEx(f, p, c) PyRun_SimpleFileExFlags(f, p, c, NULL)
#define PyRun_InteractiveOne(f, p) PyRun_InteractiveOneFlags(f, p, NULL)
#define PyRun_InteractiveLoop(f, p) PyRun_InteractiveLoopFlags(f, p, NULL)
#define PyRun_File(fp, p, s, g, l) \
    PyRun_FileExFlags(fp, p, s, g, l, 0, NULL)
#define PyRun_FileEx(fp, p, s, g, l, c) \
    PyRun_FileExFlags(fp, p, s, g, l, c, NULL)
#define PyRun_FileFlags(fp, p, s, g, l, flags) \
    PyRun_FileExFlags(fp, p, s, g, l, 0, flags)


/* Stuff with no proper home (yet) */
PyAPI_FUNC(char *) PyOS_Readline(FILE *, FILE *, const char *);
PyAPI_DATA(PyThreadState*) _PyOS_ReadlineTState;
PyAPI_DATA(char) *(*PyOS_ReadlineFunctionPointer)(FILE *, FILE *, const char *);
