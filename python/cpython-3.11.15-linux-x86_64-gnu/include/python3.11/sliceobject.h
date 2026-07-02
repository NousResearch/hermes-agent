#ifndef Py_SLICEOBJECT_H
#define Py_SLICEOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

/* The unique ellipsis object "..." */

PyAPI_DATA(PyObject) _Py_EllipsisObject; /* Don't use this directly */

#define Py_Ellipsis (&_Py_EllipsisObject)

/* Slice object interface */

/*

A slice object containing start, stop, and step data members (the
names are from range).  After much talk with Guido, it was decided to
let these be any arbitrary python type.  Py_None stands for omitted values.
*/
#ifndef Py_LIMITED_API
typedef struct {
    PyObject_HEAD
    PyObject *start, *stop, *step;      /* not NULL */
} PySliceObject;
#endif

PyAPI_DATA(PyTypeObject) PySlice_Type;
PyAPI_DATA(PyTypeObject) PyEllipsis_Type;

#define PySlice_Check(op) Py_IS_TYPE(op, &PySlice_Type)

PyAPI_FUNC(PyObject *) PySlice_New(PyObject* start, PyObject* stop,
                                  PyObject* step);
#ifndef Py_LIMITED_API
PyAPI_FUNC(PyObject *) _PySlice_FromIndices(Py_ssize_t start, Py_ssize_t stop);
PyAPI_FUNC(int) _PySlice_GetLongIndices(PySliceObject *self, PyObject *length,
                                 PyObject **start_ptr, PyObject **stop_ptr,
                                 PyObject **step_ptr);
#endif
PyAPI_FUNC(int) PySlice_GetIndices(PyObject *r, Py_ssize_t length,
                                  Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step);
Py_DEPRECATED(3.7)
PyAPI_FUNC(int) PySlice_GetIndicesEx(PyObject *r, Py_ssize_t length,
                                     Py_ssize_t *start, Py_ssize_t *stop,
                                     Py_ssize_t *step,
                                     Py_ssize_t *slicelength);

#if !defined(Py_LIMITED_API) || (Py_LIMITED_API+0 >= 0x03050400 && Py_LIMITED_API+0 < 0x03060000) || Py_LIMITED_API+0 >= 0x03060100
#define PySlice_GetIndicesEx(slice, length, start, stop, step, slicelen) (  \
    PySlice_Unpack((slice), (start), (stop), (step)) < 0 ?                  \
    ((*(slicelen) = 0), -1) :                                               \
    ((*(slicelen) = PySlice_AdjustIndices((length), (start), (stop), *(step))), \
     0))
PyAPI_FUNC(int) PySlice_Unpack(PyObject *slice,
                               Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step);
PyAPI_FUNC(Py_ssize_t) PySlice_AdjustIndices(Py_ssize_t length,
                                             Py_ssize_t *start, Py_ssize_t *stop,
                                             Py_ssize_t step);
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_SLICEOBJECT_H */
