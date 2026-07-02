#ifndef Py_CPYTHON_TUPLEOBJECT_H
#  error "this header file must not be included directly"
#endif

typedef struct {
    PyObject_VAR_HEAD
    /* ob_item contains space for 'ob_size' elements.
       Items must normally not be NULL, except during construction when
       the tuple is not yet visible outside the function that builds it. */
    PyObject *ob_item[1];
} PyTupleObject;

PyAPI_FUNC(int) _PyTuple_Resize(PyObject **, Py_ssize_t);
PyAPI_FUNC(void) _PyTuple_MaybeUntrack(PyObject *);

/* Cast argument to PyTupleObject* type. */
#define _PyTuple_CAST(op) \
    (assert(PyTuple_Check(op)), _Py_CAST(PyTupleObject*, (op)))

// Macros and static inline functions, trading safety for speed

static inline Py_ssize_t PyTuple_GET_SIZE(PyObject *op) {
    PyTupleObject *tuple = _PyTuple_CAST(op);
    return Py_SIZE(tuple);
}
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 < 0x030b0000
#  define PyTuple_GET_SIZE(op) PyTuple_GET_SIZE(_PyObject_CAST(op))
#endif

#define PyTuple_GET_ITEM(op, index) (_PyTuple_CAST(op)->ob_item[index])

/* Function *only* to be used to fill in brand new tuples */
static inline void
PyTuple_SET_ITEM(PyObject *op, Py_ssize_t index, PyObject *value) {
    PyTupleObject *tuple = _PyTuple_CAST(op);
    tuple->ob_item[index] = value;
}
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 < 0x030b0000
#define PyTuple_SET_ITEM(op, index, value) \
    PyTuple_SET_ITEM(_PyObject_CAST(op), index, _PyObject_CAST(value))
#endif

PyAPI_FUNC(void) _PyTuple_DebugMallocStats(FILE *out);
