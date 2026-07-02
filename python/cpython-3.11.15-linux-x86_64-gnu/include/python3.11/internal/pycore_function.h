#ifndef Py_INTERNAL_FUNCTION_H
#define Py_INTERNAL_FUNCTION_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

extern PyFunctionObject* _PyFunction_FromConstructor(PyFrameConstructor *constr);

extern uint32_t _PyFunction_GetVersionForCurrentState(PyFunctionObject *func);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_FUNCTION_H */
