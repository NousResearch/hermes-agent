
/* Named tuple object interface */

#ifndef Py_STRUCTSEQ_H
#define Py_STRUCTSEQ_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct PyStructSequence_Field {
    const char *name;
    const char *doc;
} PyStructSequence_Field;

typedef struct PyStructSequence_Desc {
    const char *name;
    const char *doc;
    PyStructSequence_Field *fields;
    int n_in_sequence;
} PyStructSequence_Desc;

PyAPI_DATA(const char * const) PyStructSequence_UnnamedField;

#ifndef Py_LIMITED_API
PyAPI_FUNC(void) PyStructSequence_InitType(PyTypeObject *type,
                                           PyStructSequence_Desc *desc);
PyAPI_FUNC(int) PyStructSequence_InitType2(PyTypeObject *type,
                                           PyStructSequence_Desc *desc);
#endif
PyAPI_FUNC(PyTypeObject*) PyStructSequence_NewType(PyStructSequence_Desc *desc);

PyAPI_FUNC(PyObject *) PyStructSequence_New(PyTypeObject* type);

#ifndef Py_LIMITED_API
typedef PyTupleObject PyStructSequence;

/* Macro, *only* to be used to fill in brand new objects */
#define PyStructSequence_SET_ITEM(op, i, v) PyTuple_SET_ITEM(op, i, v)

#define PyStructSequence_GET_ITEM(op, i) PyTuple_GET_ITEM(op, i)
#endif

PyAPI_FUNC(void) PyStructSequence_SetItem(PyObject*, Py_ssize_t, PyObject*);
PyAPI_FUNC(PyObject*) PyStructSequence_GetItem(PyObject*, Py_ssize_t);

#ifdef __cplusplus
}
#endif
#endif /* !Py_STRUCTSEQ_H */
