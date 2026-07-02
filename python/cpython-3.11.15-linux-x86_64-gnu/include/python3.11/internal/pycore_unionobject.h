#ifndef Py_INTERNAL_UNIONOBJECT_H
#define Py_INTERNAL_UNIONOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

extern PyTypeObject _PyUnion_Type;
#define _PyUnion_Check(op) Py_IS_TYPE(op, &_PyUnion_Type)
extern PyObject *_Py_union_type_or(PyObject *, PyObject *);

#define _PyGenericAlias_Check(op) PyObject_TypeCheck(op, &Py_GenericAliasType)
extern PyObject *_Py_subs_parameters(PyObject *, PyObject *, PyObject *, PyObject *);
extern PyObject *_Py_make_parameters(PyObject *);
extern PyObject *_Py_union_args(PyObject *self);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_UNIONOBJECT_H */
