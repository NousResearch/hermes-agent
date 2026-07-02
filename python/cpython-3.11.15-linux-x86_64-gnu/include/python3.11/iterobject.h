#ifndef Py_ITEROBJECT_H
#define Py_ITEROBJECT_H
/* Iterators (the basic kind, over a sequence) */
#ifdef __cplusplus
extern "C" {
#endif

PyAPI_DATA(PyTypeObject) PySeqIter_Type;
PyAPI_DATA(PyTypeObject) PyCallIter_Type;
#ifdef Py_BUILD_CORE
extern PyTypeObject _PyAnextAwaitable_Type;
#endif

#define PySeqIter_Check(op) Py_IS_TYPE(op, &PySeqIter_Type)

PyAPI_FUNC(PyObject *) PySeqIter_New(PyObject *);


#define PyCallIter_Check(op) Py_IS_TYPE(op, &PyCallIter_Type)

PyAPI_FUNC(PyObject *) PyCallIter_New(PyObject *, PyObject *);

#ifdef __cplusplus
}
#endif
#endif /* !Py_ITEROBJECT_H */

