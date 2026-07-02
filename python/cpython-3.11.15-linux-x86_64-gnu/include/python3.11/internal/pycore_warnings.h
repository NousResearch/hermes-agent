#ifndef Py_INTERNAL_WARNINGS_H
#define Py_INTERNAL_WARNINGS_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

struct _warnings_runtime_state {
    /* Both 'filters' and 'onceregistry' can be set in warnings.py;
       get_warnings_attr() will reset these variables accordingly. */
    PyObject *filters;  /* List */
    PyObject *once_registry;  /* Dict */
    PyObject *default_action; /* String */
    long filters_version;
};

extern int _PyWarnings_InitState(PyInterpreterState *interp);

PyAPI_FUNC(PyObject*) _PyWarnings_Init(void);

extern void _PyErr_WarnUnawaitedCoroutine(PyObject *coro);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_WARNINGS_H */
