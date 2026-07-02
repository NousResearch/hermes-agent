#ifndef Py_CPYTHON_MODSUPPORT_H
#  error "this header file must not be included directly"
#endif

/* If PY_SSIZE_T_CLEAN is defined, each functions treats #-specifier
   to mean Py_ssize_t */
#ifdef PY_SSIZE_T_CLEAN
#define _Py_VaBuildStack                _Py_VaBuildStack_SizeT
#else
PyAPI_FUNC(PyObject *) _Py_VaBuildValue_SizeT(const char *, va_list);
PyAPI_FUNC(PyObject **) _Py_VaBuildStack_SizeT(
    PyObject **small_stack,
    Py_ssize_t small_stack_len,
    const char *format,
    va_list va,
    Py_ssize_t *p_nargs);
#endif

PyAPI_FUNC(int) _PyArg_UnpackStack(
    PyObject *const *args,
    Py_ssize_t nargs,
    const char *name,
    Py_ssize_t min,
    Py_ssize_t max,
    ...);

PyAPI_FUNC(int) _PyArg_NoKeywords(const char *funcname, PyObject *kwargs);
PyAPI_FUNC(int) _PyArg_NoKwnames(const char *funcname, PyObject *kwnames);
PyAPI_FUNC(int) _PyArg_NoPositional(const char *funcname, PyObject *args);
#define _PyArg_NoKeywords(funcname, kwargs) \
    ((kwargs) == NULL || _PyArg_NoKeywords((funcname), (kwargs)))
#define _PyArg_NoKwnames(funcname, kwnames) \
    ((kwnames) == NULL || _PyArg_NoKwnames((funcname), (kwnames)))
#define _PyArg_NoPositional(funcname, args) \
    ((args) == NULL || _PyArg_NoPositional((funcname), (args)))

PyAPI_FUNC(void) _PyArg_BadArgument(const char *, const char *, const char *, PyObject *);
PyAPI_FUNC(int) _PyArg_CheckPositional(const char *, Py_ssize_t,
                                       Py_ssize_t, Py_ssize_t);
#define _PyArg_CheckPositional(funcname, nargs, min, max) \
    ((!ANY_VARARGS(max) && (min) <= (nargs) && (nargs) <= (max)) \
     || _PyArg_CheckPositional((funcname), (nargs), (min), (max)))

PyAPI_FUNC(PyObject **) _Py_VaBuildStack(
    PyObject **small_stack,
    Py_ssize_t small_stack_len,
    const char *format,
    va_list va,
    Py_ssize_t *p_nargs);

typedef struct _PyArg_Parser {
    const char *format;
    const char * const *keywords;
    const char *fname;
    const char *custom_msg;
    int pos;            /* number of positional-only arguments */
    int min;            /* minimal number of arguments */
    int max;            /* maximal number of positional arguments */
    PyObject *kwtuple;  /* tuple of keyword parameter names */
    struct _PyArg_Parser *next;
} _PyArg_Parser;

#ifdef PY_SSIZE_T_CLEAN
#define _PyArg_ParseTupleAndKeywordsFast  _PyArg_ParseTupleAndKeywordsFast_SizeT
#define _PyArg_ParseStack  _PyArg_ParseStack_SizeT
#define _PyArg_ParseStackAndKeywords  _PyArg_ParseStackAndKeywords_SizeT
#define _PyArg_VaParseTupleAndKeywordsFast  _PyArg_VaParseTupleAndKeywordsFast_SizeT
#endif

PyAPI_FUNC(int) _PyArg_ParseTupleAndKeywordsFast(PyObject *, PyObject *,
                                                 struct _PyArg_Parser *, ...);
PyAPI_FUNC(int) _PyArg_ParseStack(
    PyObject *const *args,
    Py_ssize_t nargs,
    const char *format,
    ...);
PyAPI_FUNC(int) _PyArg_ParseStackAndKeywords(
    PyObject *const *args,
    Py_ssize_t nargs,
    PyObject *kwnames,
    struct _PyArg_Parser *,
    ...);
PyAPI_FUNC(int) _PyArg_VaParseTupleAndKeywordsFast(PyObject *, PyObject *,
                                                   struct _PyArg_Parser *, va_list);
PyAPI_FUNC(PyObject * const *) _PyArg_UnpackKeywords(
        PyObject *const *args, Py_ssize_t nargs,
        PyObject *kwargs, PyObject *kwnames,
        struct _PyArg_Parser *parser,
        int minpos, int maxpos, int minkw,
        PyObject **buf);

PyAPI_FUNC(PyObject * const *) _PyArg_UnpackKeywordsWithVararg(
        PyObject *const *args, Py_ssize_t nargs,
        PyObject *kwargs, PyObject *kwnames,
        struct _PyArg_Parser *parser,
        int minpos, int maxpos, int minkw,
        int vararg, PyObject **buf);

#define _PyArg_UnpackKeywords(args, nargs, kwargs, kwnames, parser, minpos, maxpos, minkw, buf) \
    (((minkw) == 0 && (kwargs) == NULL && (kwnames) == NULL && \
      (minpos) <= (nargs) && (nargs) <= (maxpos) && args != NULL) ? (args) : \
     _PyArg_UnpackKeywords((args), (nargs), (kwargs), (kwnames), (parser), \
                           (minpos), (maxpos), (minkw), (buf)))

PyAPI_FUNC(PyObject *) _PyModule_CreateInitialized(PyModuleDef*, int apiver);
PyAPI_FUNC(int) _PyModule_Add(PyObject *, const char *, PyObject *);

PyAPI_DATA(const char *) _Py_PackageContext;
