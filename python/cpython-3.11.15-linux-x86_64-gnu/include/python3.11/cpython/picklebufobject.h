/* PickleBuffer object. This is built-in for ease of use from third-party
 * C extensions.
 */

#ifndef Py_PICKLEBUFOBJECT_H
#define Py_PICKLEBUFOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_LIMITED_API

PyAPI_DATA(PyTypeObject) PyPickleBuffer_Type;

#define PyPickleBuffer_Check(op) Py_IS_TYPE(op, &PyPickleBuffer_Type)

/* Create a PickleBuffer redirecting to the given buffer-enabled object */
PyAPI_FUNC(PyObject *) PyPickleBuffer_FromObject(PyObject *);
/* Get the PickleBuffer's underlying view to the original object
 * (NULL if released)
 */
PyAPI_FUNC(const Py_buffer *) PyPickleBuffer_GetBuffer(PyObject *);
/* Release the PickleBuffer.  Returns 0 on success, -1 on error. */
PyAPI_FUNC(int) PyPickleBuffer_Release(PyObject *);

#endif /* !Py_LIMITED_API */

#ifdef __cplusplus
}
#endif
#endif /* !Py_PICKLEBUFOBJECT_H */
