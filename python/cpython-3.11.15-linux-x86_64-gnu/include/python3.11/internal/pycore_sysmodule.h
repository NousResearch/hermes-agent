#ifndef Py_INTERNAL_SYSMODULE_H
#define Py_INTERNAL_SYSMODULE_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

PyAPI_FUNC(int) _PySys_Audit(
    PyThreadState *tstate,
    const char *event,
    const char *argFormat,
    ...);

/* We want minimal exposure of this function, so use extern rather than
   PyAPI_FUNC() to not export the symbol. */
extern void _PySys_ClearAuditHooks(PyThreadState *tstate);

PyAPI_FUNC(int) _PySys_SetAttr(PyObject *, PyObject *);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_SYSMODULE_H */
