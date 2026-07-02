#ifndef Py_INTERNAL_GENOBJECT_H
#define Py_INTERNAL_GENOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

extern PyObject *_PyGen_yf(PyGenObject *);
extern PyObject *_PyCoro_GetAwaitableIter(PyObject *o);
extern PyObject *_PyAsyncGenValueWrapperNew(PyObject *);

/* runtime lifecycle */

extern void _PyAsyncGen_Fini(PyInterpreterState *);


/* other API */

#ifndef WITH_FREELISTS
// without freelists
#  define _PyAsyncGen_MAXFREELIST 0
#endif

#ifndef _PyAsyncGen_MAXFREELIST
#  define _PyAsyncGen_MAXFREELIST 80
#endif

struct _Py_async_gen_state {
#if _PyAsyncGen_MAXFREELIST > 0
    /* Freelists boost performance 6-10%; they also reduce memory
       fragmentation, as _PyAsyncGenWrappedValue and PyAsyncGenASend
       are short-living objects that are instantiated for every
       __anext__() call. */
    struct _PyAsyncGenWrappedValue* value_freelist[_PyAsyncGen_MAXFREELIST];
    int value_numfree;

    struct PyAsyncGenASend* asend_freelist[_PyAsyncGen_MAXFREELIST];
    int asend_numfree;
#endif
};


#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_GENOBJECT_H */
