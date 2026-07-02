#ifndef Py_INTERNAL_PARSER_H
#define Py_INTERNAL_PARSER_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

extern struct _mod* _PyParser_ASTFromString(
    const char *str,
    PyObject* filename,
    int mode,
    PyCompilerFlags *flags,
    PyArena *arena);
extern struct _mod* _PyParser_ASTFromFile(
    FILE *fp,
    PyObject *filename_ob,
    const char *enc,
    int mode,
    const char *ps1,
    const char *ps2,
    PyCompilerFlags *flags,
    int *errcode,
    PyArena *arena);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_PARSER_H */
