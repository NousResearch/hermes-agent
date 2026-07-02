#ifndef Py_ODICTOBJECT_H
#define Py_ODICTOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif


/* OrderedDict */
/* This API is optional and mostly redundant. */

#ifndef Py_LIMITED_API

typedef struct _odictobject PyODictObject;

PyAPI_DATA(PyTypeObject) PyODict_Type;
PyAPI_DATA(PyTypeObject) PyODictIter_Type;
PyAPI_DATA(PyTypeObject) PyODictKeys_Type;
PyAPI_DATA(PyTypeObject) PyODictItems_Type;
PyAPI_DATA(PyTypeObject) PyODictValues_Type;

#define PyODict_Check(op) PyObject_TypeCheck(op, &PyODict_Type)
#define PyODict_CheckExact(op) Py_IS_TYPE(op, &PyODict_Type)
#define PyODict_SIZE(op) PyDict_GET_SIZE((op))

PyAPI_FUNC(PyObject *) PyODict_New(void);
PyAPI_FUNC(int) PyODict_SetItem(PyObject *od, PyObject *key, PyObject *item);
PyAPI_FUNC(int) PyODict_DelItem(PyObject *od, PyObject *key);

/* wrappers around PyDict* functions */
#define PyODict_GetItem(od, key) PyDict_GetItem(_PyObject_CAST(od), key)
#define PyODict_GetItemWithError(od, key) \
    PyDict_GetItemWithError(_PyObject_CAST(od), key)
#define PyODict_Contains(od, key) PyDict_Contains(_PyObject_CAST(od), key)
#define PyODict_Size(od) PyDict_Size(_PyObject_CAST(od))
#define PyODict_GetItemString(od, key) \
    PyDict_GetItemString(_PyObject_CAST(od), key)

#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_ODICTOBJECT_H */
