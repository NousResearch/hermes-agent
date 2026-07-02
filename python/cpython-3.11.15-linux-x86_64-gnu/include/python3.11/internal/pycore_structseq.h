#ifndef Py_INTERNAL_STRUCTSEQ_H
#define Py_INTERNAL_STRUCTSEQ_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif


/* other API */

PyAPI_FUNC(PyTypeObject *) _PyStructSequence_NewType(
    PyStructSequence_Desc *desc,
    unsigned long tp_flags);

PyAPI_FUNC(int) _PyStructSequence_InitType(
    PyTypeObject *type,
    PyStructSequence_Desc *desc,
    unsigned long tp_flags);

extern void _PyStructSequence_FiniType(PyTypeObject *type);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_STRUCTSEQ_H */
