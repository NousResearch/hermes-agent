#ifndef Py_INTERNAL_EXCEPTIONS_H
#define Py_INTERNAL_EXCEPTIONS_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif


/* runtime lifecycle */

extern PyStatus _PyExc_InitState(PyInterpreterState *);
extern PyStatus _PyExc_InitGlobalObjects(PyInterpreterState *);
extern int _PyExc_InitTypes(PyInterpreterState *);
extern void _PyExc_Fini(PyInterpreterState *);


/* other API */

struct _Py_exc_state {
    // The dict mapping from errno codes to OSError subclasses
    PyObject *errnomap;
    PyBaseExceptionObject *memerrors_freelist;
    int memerrors_numfree;
    // The ExceptionGroup type
    PyObject *PyExc_ExceptionGroup;
};

extern void _PyExc_ClearExceptionGroupType(PyInterpreterState *);


#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_EXCEPTIONS_H */
