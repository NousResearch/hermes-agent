#ifndef Py_CPYTHON_FILEOBJECT_H
#  error "this header file must not be included directly"
#endif

PyAPI_FUNC(char *) Py_UniversalNewlineFgets(char *, int, FILE*, PyObject *);
PyAPI_FUNC(char *) _Py_UniversalNewlineFgetsWithSize(char *, int, FILE*, PyObject *, size_t*);

/* The std printer acts as a preliminary sys.stderr until the new io
   infrastructure is in place. */
PyAPI_FUNC(PyObject *) PyFile_NewStdPrinter(int);
PyAPI_DATA(PyTypeObject) PyStdPrinter_Type;

typedef PyObject * (*Py_OpenCodeHookFunction)(PyObject *, void *);

PyAPI_FUNC(PyObject *) PyFile_OpenCode(const char *utf8path);
PyAPI_FUNC(PyObject *) PyFile_OpenCodeObject(PyObject *path);
PyAPI_FUNC(int) PyFile_SetOpenCodeHook(Py_OpenCodeHookFunction hook, void *userData);

PyAPI_FUNC(int) _PyLong_FileDescriptor_Converter(PyObject *, void *);
