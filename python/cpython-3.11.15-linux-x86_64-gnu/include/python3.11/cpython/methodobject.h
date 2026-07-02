#ifndef Py_CPYTHON_METHODOBJECT_H
#  error "this header file must not be included directly"
#endif

// PyCFunctionObject structure

typedef struct {
    PyObject_HEAD
    PyMethodDef *m_ml; /* Description of the C function to call */
    PyObject    *m_self; /* Passed as 'self' arg to the C func, can be NULL */
    PyObject    *m_module; /* The __module__ attribute, can be anything */
    PyObject    *m_weakreflist; /* List of weak references */
    vectorcallfunc vectorcall;
} PyCFunctionObject;

#define _PyCFunctionObject_CAST(func) \
    (assert(PyCFunction_Check(func)), \
     _Py_CAST(PyCFunctionObject*, (func)))


// PyCMethodObject structure

typedef struct {
    PyCFunctionObject func;
    PyTypeObject *mm_class; /* Class that defines this method */
} PyCMethodObject;

#define _PyCMethodObject_CAST(func) \
    (assert(PyCMethod_Check(func)), \
     _Py_CAST(PyCMethodObject*, (func)))

PyAPI_DATA(PyTypeObject) PyCMethod_Type;

#define PyCMethod_CheckExact(op) Py_IS_TYPE(op, &PyCMethod_Type)
#define PyCMethod_Check(op) PyObject_TypeCheck(op, &PyCMethod_Type)


/* Static inline functions for direct access to these values.
   Type checks are *not* done, so use with care. */
static inline PyCFunction PyCFunction_GET_FUNCTION(PyObject *func) {
    return _PyCFunctionObject_CAST(func)->m_ml->ml_meth;
}
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 < 0x030b0000
#  define PyCFunction_GET_FUNCTION(func) PyCFunction_GET_FUNCTION(_PyObject_CAST(func))
#endif

static inline PyObject* PyCFunction_GET_SELF(PyObject *func_obj) {
    PyCFunctionObject *func = _PyCFunctionObject_CAST(func_obj);
    if (func->m_ml->ml_flags & METH_STATIC) {
        return _Py_NULL;
    }
    return func->m_self;
}
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 < 0x030b0000
#  define PyCFunction_GET_SELF(func) PyCFunction_GET_SELF(_PyObject_CAST(func))
#endif

static inline int PyCFunction_GET_FLAGS(PyObject *func) {
    return _PyCFunctionObject_CAST(func)->m_ml->ml_flags;
}
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 < 0x030b0000
#  define PyCFunction_GET_FLAGS(func) PyCFunction_GET_FLAGS(_PyObject_CAST(func))
#endif

static inline PyTypeObject* PyCFunction_GET_CLASS(PyObject *func_obj) {
    PyCFunctionObject *func = _PyCFunctionObject_CAST(func_obj);
    if (func->m_ml->ml_flags & METH_METHOD) {
        return _PyCMethodObject_CAST(func)->mm_class;
    }
    return _Py_NULL;
}
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 < 0x030b0000
#  define PyCFunction_GET_CLASS(func) PyCFunction_GET_CLASS(_PyObject_CAST(func))
#endif
