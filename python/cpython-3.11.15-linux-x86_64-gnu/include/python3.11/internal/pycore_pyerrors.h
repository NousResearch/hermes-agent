#ifndef Py_INTERNAL_PYERRORS_H
#define Py_INTERNAL_PYERRORS_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif


/* runtime lifecycle */

extern PyStatus _PyErr_InitTypes(PyInterpreterState *);
extern void _PyErr_FiniTypes(PyInterpreterState *);


/* other API */

static inline PyObject* _PyErr_Occurred(PyThreadState *tstate)
{
    assert(tstate != NULL);
    return tstate->curexc_type;
}

static inline void _PyErr_ClearExcState(_PyErr_StackItem *exc_state)
{
    Py_CLEAR(exc_state->exc_value);
}

PyAPI_FUNC(PyObject*) _PyErr_StackItemToExcInfoTuple(
    _PyErr_StackItem *err_info);

PyAPI_FUNC(void) _PyErr_Fetch(
    PyThreadState *tstate,
    PyObject **type,
    PyObject **value,
    PyObject **traceback);

PyAPI_FUNC(int) _PyErr_ExceptionMatches(
    PyThreadState *tstate,
    PyObject *exc);

PyAPI_FUNC(void) _PyErr_Restore(
    PyThreadState *tstate,
    PyObject *type,
    PyObject *value,
    PyObject *traceback);

PyAPI_FUNC(void) _PyErr_SetObject(
    PyThreadState *tstate,
    PyObject *type,
    PyObject *value);

PyAPI_FUNC(void) _PyErr_ChainStackItem(
    _PyErr_StackItem *exc_info);

PyAPI_FUNC(void) _PyErr_Clear(PyThreadState *tstate);

PyAPI_FUNC(void) _PyErr_SetNone(PyThreadState *tstate, PyObject *exception);

PyAPI_FUNC(PyObject *) _PyErr_NoMemory(PyThreadState *tstate);

PyAPI_FUNC(void) _PyErr_SetString(
    PyThreadState *tstate,
    PyObject *exception,
    const char *string);

PyAPI_FUNC(PyObject *) _PyErr_Format(
    PyThreadState *tstate,
    PyObject *exception,
    const char *format,
    ...);

PyAPI_FUNC(void) _PyErr_NormalizeException(
    PyThreadState *tstate,
    PyObject **exc,
    PyObject **val,
    PyObject **tb);

PyAPI_FUNC(PyObject *) _PyErr_FormatFromCauseTstate(
    PyThreadState *tstate,
    PyObject *exception,
    const char *format,
    ...);

PyAPI_FUNC(PyObject *) _PyExc_CreateExceptionGroup(
    const char *msg,
    PyObject *excs);

PyAPI_FUNC(PyObject *) _PyExc_PrepReraiseStar(
    PyObject *orig,
    PyObject *excs);

PyAPI_FUNC(int) _PyErr_CheckSignalsTstate(PyThreadState *tstate);

PyAPI_FUNC(void) _Py_DumpExtensionModules(int fd, PyInterpreterState *interp);

extern PyObject* _Py_Offer_Suggestions(PyObject* exception);
PyAPI_FUNC(Py_ssize_t) _Py_UTF8_Edit_Cost(PyObject *str_a, PyObject *str_b,
                                          Py_ssize_t max_cost);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_PYERRORS_H */
