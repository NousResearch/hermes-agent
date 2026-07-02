/* Generator object interface */

#ifndef Py_LIMITED_API
#ifndef Py_GENOBJECT_H
#define Py_GENOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

/* --- Generators --------------------------------------------------------- */

/* _PyGenObject_HEAD defines the initial segment of generator
   and coroutine objects. */
#define _PyGenObject_HEAD(prefix)                                           \
    PyObject_HEAD                                                           \
    /* The code object backing the generator */                             \
    PyCodeObject *prefix##_code;                                            \
    /* List of weak reference. */                                           \
    PyObject *prefix##_weakreflist;                                         \
    /* Name of the generator. */                                            \
    PyObject *prefix##_name;                                                \
    /* Qualified name of the generator. */                                  \
    PyObject *prefix##_qualname;                                            \
    _PyErr_StackItem prefix##_exc_state;                                    \
    PyObject *prefix##_origin_or_finalizer;                                 \
    char prefix##_hooks_inited;                                             \
    char prefix##_closed;                                                   \
    char prefix##_running_async;                                            \
    /* The frame */                                                         \
    int8_t prefix##_frame_state;                                            \
    PyObject *prefix##_iframe[1];

typedef struct {
    /* The gi_ prefix is intended to remind of generator-iterator. */
    _PyGenObject_HEAD(gi)
} PyGenObject;

PyAPI_DATA(PyTypeObject) PyGen_Type;

#define PyGen_Check(op) PyObject_TypeCheck(op, &PyGen_Type)
#define PyGen_CheckExact(op) Py_IS_TYPE(op, &PyGen_Type)

PyAPI_FUNC(PyObject *) PyGen_New(PyFrameObject *);
PyAPI_FUNC(PyObject *) PyGen_NewWithQualName(PyFrameObject *,
    PyObject *name, PyObject *qualname);
PyAPI_FUNC(int) _PyGen_SetStopIterationValue(PyObject *);
PyAPI_FUNC(int) _PyGen_FetchStopIterationValue(PyObject **);
PyAPI_FUNC(void) _PyGen_Finalize(PyObject *self);


/* --- PyCoroObject ------------------------------------------------------- */

typedef struct {
    _PyGenObject_HEAD(cr)
} PyCoroObject;

PyAPI_DATA(PyTypeObject) PyCoro_Type;
PyAPI_DATA(PyTypeObject) _PyCoroWrapper_Type;

#define PyCoro_CheckExact(op) Py_IS_TYPE(op, &PyCoro_Type)
PyAPI_FUNC(PyObject *) PyCoro_New(PyFrameObject *,
    PyObject *name, PyObject *qualname);


/* --- Asynchronous Generators -------------------------------------------- */

typedef struct {
    _PyGenObject_HEAD(ag)
} PyAsyncGenObject;

PyAPI_DATA(PyTypeObject) PyAsyncGen_Type;
PyAPI_DATA(PyTypeObject) _PyAsyncGenASend_Type;
PyAPI_DATA(PyTypeObject) _PyAsyncGenWrappedValue_Type;
PyAPI_DATA(PyTypeObject) _PyAsyncGenAThrow_Type;

PyAPI_FUNC(PyObject *) PyAsyncGen_New(PyFrameObject *,
    PyObject *name, PyObject *qualname);

#define PyAsyncGen_CheckExact(op) Py_IS_TYPE(op, &PyAsyncGen_Type)


#undef _PyGenObject_HEAD

#ifdef __cplusplus
}
#endif
#endif /* !Py_GENOBJECT_H */
#endif /* Py_LIMITED_API */
